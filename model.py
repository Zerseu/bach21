"""Training and sampling of a token model for one composer and instrument.

Use the command-line scripts rather than this module directly:

    python train.py bach violin                  # train bach21data/bach/violin/note_model.torch
    python generate.py bach violin --length 1000 # write a 1000-note piece as MIDI and MusicXML

Running ``python model.py bach violin`` still does both steps, as it did before those scripts existed.
"""
import json
import os
import random
import sys
from typing import Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import Config, log, seed_everything
from data import generate_input, generate_output, get_dir
from motifs import motif_continuation, query_all_motifs
from network import DEVICE, TorchModule
from sequences import (IGNORE_INDEX, build_vocabulary, extra_sentences, generate_xy, read_sentences, split_data,
                       transpose_sentences)

TRAIN_SEED = 0          # fixes the part order, the train/validation split, initial weights and batch order
TRAIN_RATIO = 0.8       # share of the tokens (in whole parts) used for training; the rest is validation
DEFAULT_LENGTH = 1000   # notes written by generate()


class TorchDataset(Dataset):
    """Input and target windows, kept on the training device."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert len(x) == len(y)
        self.x = torch.from_numpy(x).long().to(DEVICE)
        self.y = torch.from_numpy(y).long().to(DEVICE)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class Worker:
    """Everything for one model: its data folder, vocabulary, training and sampling.

    kind is a section of config.ini and the prefix of its files: '<kind>_input.txt' (training tokens),
    '<kind>_model.torch' (trained weights), '<kind>_vocabulary.json', '<kind>_log.csv' (per-epoch losses),
    '<kind>_output.txt' (generated tokens).
    """

    def __init__(self, composer: str, instruments: list[str], kind: str = 'note', config: Optional[Config] = None):
        self.composer = composer
        self.instruments = instruments
        self.kind = kind
        self.c = (config or Config())[kind]
        self.crt_dir = get_dir(composer, instruments)
        self.map_direct: dict[str, int] = {}
        self.map_reverse: dict[int, str] = {}
        self.model: Optional[TorchModule] = None

    def path(self, suffix: str) -> str:
        """Path of one of this model's files, for example path('model.torch') -> .../note_model.torch."""
        return os.path.join(self.crt_dir, f'{self.kind}_{suffix}')

    @property
    def vocabulary_size(self) -> int:
        """Number of distinct tokens the model knows (its output size)."""
        return len(self.map_direct)

    # ----- data -----

    def training_sentences(self) -> tuple[list[list[str]], list[list[str]]]:
        """Training and validation parts.

        Validation is the last ~20% of the (shuffled) parts, untransposed. Training parts are also used
        transposed, and can be extended with the composer's other parts that fit the instrument range.
        """
        sentences = read_sentences(self.path('input.txt'))
        sentences_trn, sentences_val = split_data(sentences, TRAIN_RATIO)
        pitched = self.kind in ('pitch', 'note')
        if self.c['transpose'] > 0 and pitched:
            sentences_trn = transpose_sentences(sentences_trn, sentences, self.c['transpose'], self.kind)
            log(f"Transposition: {len(sentences_trn)} training parts (shifts up to +/-{self.c['transpose']} semitones)")
        if self.c['extra_training_data'] and len(self.instruments) > 0 and pitched:
            extra = extra_sentences(self.composer, self.kind, sentences, sentences_trn + sentences_val)
            log(f'Extra training data: {len(extra)} parts of other {self.composer} instruments, '
                f'{sum(map(len, extra))} tokens')
            if self.c['transpose_extra'] > 0:
                extra = transpose_sentences(extra, sentences, self.c['transpose_extra'], self.kind)
                log(f'Extra training data after transposition: {len(extra)} parts')
            sentences_trn = sentences_trn + extra
        return sentences_trn, sentences_val

    def _set_vocabulary(self, map_direct: dict[str, int]):
        """Sets the token -> id map and its inverse."""
        self.map_direct = map_direct
        self.map_reverse = {idx: word for word, idx in map_direct.items()}

    def load_vocabulary(self):
        """Reads the vocabulary saved with the model, or rebuilds it from the training data for models
        trained before the vocabulary was saved."""
        if os.path.exists(self.path('vocabulary.json')):
            with open(self.path('vocabulary.json'), 'rt') as file:
                self._set_vocabulary({word: idx for idx, word in enumerate(json.load(file))})
        else:
            sentences_trn, sentences_val = self.training_sentences()
            self._set_vocabulary(build_vocabulary(sentences_trn + sentences_val))

    def motifs(self) -> dict[str, int]:
        """Frequent motifs of the input file, computed once and cached in '<kind>_motifs.json'."""
        pth = self.path('motifs.json')
        if not os.path.exists(pth):
            motifs = query_all_motifs(self.path('input.txt'), self.map_direct, self.map_reverse,
                                      self.kind in ('pitch', 'note'))
            with open(pth, 'wt') as file:
                json.dump(motifs, file, indent=4)
        with open(pth, 'rt') as file:
            return json.load(file)

    # ----- model -----

    def new_model(self, lora_enable: bool) -> TorchModule:
        """A freshly initialised network sized for the current vocabulary and built from the config.

        Training creates it without LoRA and adds the adapters after loading base_model; loading a
        trained LoRA model needs them at once (lora_enable=True), so the saved keys match.
        """
        c = self.c
        return TorchModule(self.vocabulary_size, self.map_direct, self.map_reverse, c['hidden_size'],
                           lora_enable=lora_enable,
                           lora_r=c['lora_r'],
                           lora_alpha=c['lora_alpha'],
                           lora_dropout=c['lora_dropout'],
                           lora_targets=c['lora_targets'],
                           add_causal_attn=c['add_causal_attn'],
                           attn_heads=c['attn_heads'],
                           dropout=c['dropout'],
                           output_dropout=c['output_dropout'],
                           tie_weights=c['tie_weights'])

    def train(self, retrain: bool = False) -> str:
        """Trains the model and returns the path of '<kind>_model.torch'.

        An existing model is kept unless retrain is True. The best epoch (lowest validation loss) is the
        one saved, and the model file only appears once training has finished, so an interrupted run
        leaves any previous model in place.
        """
        pth_model = self.path('model.torch')
        if os.path.exists(pth_model) and not retrain:
            log(f'{pth_model} already exists, skipping training (use --retrain to train again).')
            return pth_model
        c = self.c

        # The vocabulary covers training (with transpositions and extra parts) and validation, so every
        # validation token has an id.
        sentences_trn, sentences_val = self.training_sentences()
        self._set_vocabulary(build_vocabulary(sentences_trn + sentences_val))
        log(f'Vocabulary: {self.vocabulary_size} tokens')
        data_trn = [[self.map_direct[word] for word in sentence] for sentence in sentences_trn]
        data_val = [[self.map_direct[word] for word in sentence] for sentence in sentences_val]
        if c['motif_threshold'] > 0:
            self.motifs()   # build the motif cache now, while the vocabulary matches the training data

        window_stride = c['window_stride'] or max(1, c['number_of_steps'] // 4)
        d_trn = TorchDataset(*generate_xy(data_trn, c['number_of_steps'], window_stride))
        d_val = TorchDataset(*generate_xy(data_val, c['number_of_steps'], window_stride))
        log(f'Windows: {len(d_trn)} training, {len(d_val)} validation')

        model = self.new_model(lora_enable=False)
        if c['base_model']:
            log(f"Loading base model {c['base_model']}...")
            model.load_state_dict(torch.load(c['base_model'], map_location=DEVICE))
        if c['lora_enable']:
            if not c['base_model']:
                log('[WARN] LoRA is enabled without a base_model, so the frozen base weights stay random.')
            model.apply_lora(c['lora_r'], c['lora_alpha'], c['lora_dropout'], c['lora_targets'])
            if c['lora_peft_only']:
                from lora import mark_trainable_lora_only
                mark_trainable_lora_only(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(f'Trainable params: {trainable:,} / {total:,}')

        optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                          lr=c['learning_rate'], weight_decay=c['weight_decay'])
        # The best weights so far are kept in '<kind>_model_best.torch' during training and moved to
        # '<kind>_model.torch' at the end.
        pth_best = self.path('model_best.torch')
        train_model(model,
                    DataLoader(dataset=d_trn, batch_size=c['batch_size'], shuffle=True),
                    DataLoader(dataset=d_val, batch_size=c['batch_size']),
                    optimizer,
                    self.path('log.csv'),
                    pth_best,
                    c['number_of_epochs'],
                    c['early_stopping_patience'],
                    c['grad_clip'],
                    c['lr_decay'])
        with open(self.path('vocabulary.json'), 'wt') as file:
            json.dump([self.map_reverse[idx] for idx in range(self.vocabulary_size)], file)
        os.replace(pth_best, pth_model)
        log(f'Saved {pth_model}')
        return pth_model

    def load(self):
        """Loads the trained model for sampling (on the CPU)."""
        pth_model = self.path('model.torch')
        if not os.path.exists(pth_model):
            raise FileNotFoundError(f'{pth_model} does not exist. Train the model first (python train.py '
                                    f'{self.composer} {" ".join(self.instruments)}).')
        self.load_vocabulary()
        self.model = self.new_model(lora_enable=self.c['lora_enable'])
        self.model.load_state_dict(torch.load(pth_model, map_location=DEVICE))
        self.model = self.model.cpu()
        self.model.eval()

    def generate(self, length: int = DEFAULT_LENGTH, seed: int = 0, keep_prompt: bool = False) -> str:
        """Samples `length` new tokens and writes them to '<kind>_output.txt'; returns its path.

        The model is primed with number_of_steps consecutive tokens from a random place in the training
        file (chosen by `seed`). The prompt is real music, so it is only written when keep_prompt is True.
        """
        if self.model is None:
            self.load()
        seed_everything(seed)
        number_of_steps = self.c['number_of_steps']
        # The prompt is taken from the whole input file joined into one stream, so it can span the end of
        # one part and the start of the next.
        with open(self.path('input.txt'), 'rt') as file:
            content = file.read().split()
        start = random.randrange(0, len(content) - number_of_steps)
        prompt = content[start:start + number_of_steps]
        sentence_ids = [self.map_direct[word] for word in prompt]
        motifs = self.motifs() if self.c['motif_threshold'] > 0 else {}

        generated = []
        with torch.no_grad():
            for _ in tqdm(range(length), desc='Generating'):
                # A sliding context of the last number_of_steps tokens, the window length used in training.
                token = self.next_token(sentence_ids[-number_of_steps:], motifs)
                sentence_ids.append(token)
                generated.append(self.map_reverse[token])

        pth_output = self.path('output.txt')
        with open(pth_output, 'wt') as file:
            file.write(' '.join(prompt + generated if keep_prompt else generated))
        return pth_output

    def next_token(self, seq: list[int], motifs: dict[str, int]) -> int:
        """Samples the next token id. When the model is unsure (top probability at or below
        motif_threshold) a matching corpus motif is continued instead, if there is one."""
        with torch.no_grad():
            # Only the prediction after the last token is needed.
            logits = self.model(torch.tensor([seq], dtype=torch.long))[:, -1, :]
        prob_max, token = sample(logits, self.c['temperature'], self.c['top_p'])
        if prob_max <= self.c['motif_threshold']:
            continuation = motif_continuation(motifs, self.map_direct, seq)
            if continuation is not None:
                return continuation
        return token


def sample(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> tuple[float, int]:
    """Temperature sampling from logits of shape (1, vocabulary), optionally restricted to the smallest set
    of tokens whose probabilities add up to top_p (nucleus sampling). Returns (top probability, token id)."""
    # Softmax in float64 (subtracting the maximum keeps exp from overflowing); temperatures below 1 make
    # the distribution sharper, above 1 flatter. prob_max is measured after the temperature is applied.
    pred = logits.detach().numpy().astype(np.float64)[0] / temperature
    pred = np.exp(pred - np.max(pred))
    pred = pred / np.sum(pred)
    prob_max = float(np.max(pred))
    if top_p < 1.0:
        order = np.argsort(-pred)
        keep = order[:int(np.searchsorted(np.cumsum(pred[order]), top_p)) + 1]
        nucleus = np.zeros_like(pred)
        nucleus[keep] = pred[keep]
        pred = nucleus / np.sum(nucleus)
    return prob_max, int(np.random.choice(len(pred), p=pred))


def train_model(model: Module,
                train_loader: DataLoader,
                valid_loader: DataLoader,
                optimizer: AdamW,
                csv_logger: str,
                pth_best: str,
                num_epochs: int = 10,
                patience: int = 0,
                grad_clip: float = 0.0,
                lr_decay: float = 0.0):
    """The training loop. Losses are per-token averages over all non-padding positions.

    After each epoch the weights are saved to pth_best if the validation loss improved. Otherwise, with
    0 < lr_decay < 1, the best weights are reloaded and the learning rate is multiplied by lr_decay.
    Training stops after `patience` epochs in a row without improvement (0 means never early).
    One line per epoch goes to csv_logger: epoch, train_loss, validation_loss, best.
    """
    # Summed, not averaged, per batch so that the epoch loss can be divided by the true number of tokens;
    # each batch's gradient still uses the per-token mean.
    loss_function = CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='sum')
    best_valid_loss = float('inf')
    epochs_without_improvement = 0
    with open(csv_logger, 'wt') as file:
        file.write('epoch,train_loss,validation_loss,best\n')
        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            model.train()
            total_train_loss = 0.0
            total_train_tokens = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                tokens = int((labels != IGNORE_INDEX).sum())
                loss = loss_function(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
                (loss / max(tokens, 1)).backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                total_train_loss += loss.item()
                total_train_tokens += tokens

            model.eval()
            total_valid_loss = 0.0
            total_valid_tokens = 0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    outputs = model(inputs)
                    loss = loss_function(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
                    total_valid_loss += loss.item()
                    total_valid_tokens += int((labels != IGNORE_INDEX).sum())

            avg_train_loss = total_train_loss / max(total_train_tokens, 1)
            avg_valid_loss = total_valid_loss / max(total_valid_tokens, 1)
            improved = avg_valid_loss < best_valid_loss
            if improved:
                best_valid_loss = avg_valid_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), pth_best)
            else:
                epochs_without_improvement += 1
                if 0 < lr_decay < 1:
                    model.load_state_dict(torch.load(pth_best, map_location=DEVICE))
                    for group in optimizer.param_groups:
                        group['lr'] *= lr_decay
            file.write(f'{epoch + 1},{avg_train_loss:.4f},{avg_valid_loss:.4f},{int(improved)}\n')
            file.flush()
            log(f'Epoch {epoch + 1}: train {avg_train_loss:.4f}, validation {avg_valid_loss:.4f}'
                f'{" (best)" if improved else ""}')
            if 0 < patience <= epochs_without_improvement:
                break


def main_train(composer: str, instruments: list[str], kind: str = 'note', retrain: bool = False) -> str:
    """Builds the token files if needed and trains the model; returns the model path.

    Seeding first makes the part order, the split, the initial weights and the batch order the same on
    every run with the same settings.
    """
    seed_everything(TRAIN_SEED)
    generate_input(composer, instruments)
    return Worker(composer, instruments, kind).train(retrain=retrain)


def main_generate(composer: str, instruments: list[str], kind: str = 'note', length: int = DEFAULT_LENGTH,
                  seed: int = 0, keep_prompt: bool = False) -> tuple[str, str, str]:
    """Generates a piece and writes it as text, MIDI and MusicXML; returns the three paths."""
    pth_text = Worker(composer, instruments, kind).generate(length=length, seed=seed, keep_prompt=keep_prompt)
    pth_midi, pth_xml = generate_output(composer, instruments, kind)
    return pth_text, pth_midi, pth_xml


if __name__ == '__main__':
    # Train (if no model exists yet) and then generate DEFAULT_LENGTH notes with seed 0.
    if len(sys.argv) < 2:
        sys.exit('Usage: python model.py COMPOSER [INSTRUMENT ...]')
    main_train(sys.argv[1], sys.argv[2:])
    main_generate(sys.argv[1], sys.argv[2:])

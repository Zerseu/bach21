"""Training and sampling of a token model for one composer and instrument.

Use the command-line scripts rather than this module directly:

    python train.py bach violin                  # train bach21data/bach/violin/note_model.torch
    python generate.py bach violin --length 1000 # write a 1000-note piece as MIDI and MusicXML

Running ``python model.py bach violin`` still does both steps, as it did before those scripts existed.
"""
import json
import math
import os
import random
import sys
from typing import Iterable, Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import Config, log, seed_everything
from data import REST_TOKEN, generate_input, generate_output, get_dir, split_note_token
from motifs import motif_continuation, query_all_motifs
from network import DEVICE, TorchModule
from sequences import (IGNORE_INDEX, RandomWindows, build_vocabulary, extra_sentences, generate_xy,
                       note_token_features, read_sentences, split_data, transpose_sentences)

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
        self._pitch_rows: Optional[tuple[np.ndarray, int]] = None   # see pitch_rows(); built on first use

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
        transposed, and can be extended with parts of the composer's other instruments, and of the other
        composers named in extra_composers, that fit the instrument range (see extra_sentences).
        """
        sentences = read_sentences(self.path('input.txt'))
        sentences_trn, sentences_val = split_data(sentences, TRAIN_RATIO)
        pitched = self.kind in ('pitch', 'note')
        if self.c['transpose'] > 0 and pitched:
            sentences_trn = transpose_sentences(sentences_trn, sentences, self.c['transpose'], self.kind)
            log(f"Transposition: {len(sentences_trn)} training parts (shifts up to +/-{self.c['transpose']} semitones)")
        if self.c['extra_training_data'] and pitched:
            # The allowed tokens are fixed before any extra part is added, so every source is filtered
            # against the same vocabulary: the instrument's own parts with their transpositions.
            vocabulary = sentences_trn + sentences_val
            extra = []
            if len(self.instruments) > 0:
                # Without an instrument the model already trains on every part of the composer.
                own = extra_sentences(self.composer, self.kind, sentences, vocabulary)
                log(f'Extra training data: {len(own)} parts of other {self.composer} instruments, '
                    f'{sum(map(len, own))} tokens')
                extra += own
            for composer in self.extra_composers():
                other = extra_sentences(composer, self.kind, sentences, vocabulary)
                log(f'Extra training data: {len(other)} parts of {composer}, {sum(map(len, other))} tokens')
                extra += other
            if self.c['transpose_extra'] > 0 and extra:
                extra = transpose_sentences(extra, sentences, self.c['transpose_extra'], self.kind)
                log(f'Extra training data after transposition: {len(extra)} parts')
            sentences_trn = sentences_trn + extra
        return sentences_trn, sentences_val

    def extra_composers(self) -> list[str]:
        """The other composers whose parts are added to the training data: config.ini's extra_composers
        (comma-separated, lower case), without the model's own composer."""
        names = [name.strip().lower() for name in self.c['extra_composers'].split(',')]
        return [name for name in names if name and name != self.composer.lower()]

    def _set_vocabulary(self, map_direct: dict[str, int]):
        """Sets the token -> id map and its inverse."""
        self.map_direct = map_direct
        self.map_reverse = {idx: word for word, idx in map_direct.items()}
        self._pitch_rows = None   # belongs to the previous vocabulary

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
        # The music-aware options read pitch and duration out of the tokens, which only the combined
        # 'note' tokens have. For the experimental 'pitch' and 'duration' kinds they are switched off.
        wanted = [key for key in ('structured_embedding', 'pitch_relations', 'pattern_pointers') if c[key]]
        if wanted and self.kind != 'note':
            log(f"[WARN] {', '.join(wanted)} only apply to the note model; ignored for [{self.kind}].")
            wanted = []
        token_features = None
        if wanted:
            token_features = note_token_features([self.map_reverse[idx] for idx in range(self.vocabulary_size)])
        return TorchModule(self.vocabulary_size, self.map_direct, self.map_reverse, c['hidden_size'],
                           token_features=token_features,
                           structured_embedding='structured_embedding' in wanted,
                           pitch_relations='pitch_relations' in wanted,
                           pattern_pointers='pattern_pointers' in wanted,
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

        # Validation always uses the same fixed windows (one every window_stride tokens), so its loss is
        # comparable between runs whatever the training windows are.
        window_stride = c['window_stride'] or max(1, c['number_of_steps'] // 4)
        d_val = TorchDataset(*generate_xy(data_val, c['number_of_steps'], window_stride))
        if c['random_windows']:
            # Windows at random starts, redrawn for every batch; an epoch is about one visit per token.
            batches_trn = RandomWindows(data_trn, c['number_of_steps'], c['batch_size'], seed=TRAIN_SEED,
                                        device=DEVICE)
            log(f'Windows: random, {len(batches_trn)} batches of {c["batch_size"]} per epoch; '
                f'{len(d_val)} validation')
        else:
            # The same fixed windows every epoch, in a new order each time.
            d_trn = TorchDataset(*generate_xy(data_trn, c['number_of_steps'], window_stride))
            batches_trn = DataLoader(dataset=d_trn, batch_size=c['batch_size'], shuffle=True)
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
        scheduler = make_scheduler(optimizer, c['lr_schedule'], c['lr_warmup_steps'],
                                   total_steps=c['number_of_epochs'] * len(batches_trn))
        # The best weights so far are kept in '<kind>_model_best.torch' during training and moved to
        # '<kind>_model.torch' at the end.
        pth_best = self.path('model_best.torch')
        train_model(model,
                    batches_trn,
                    DataLoader(dataset=d_val, batch_size=c['batch_size']),
                    optimizer,
                    self.path('log.csv'),
                    pth_best,
                    c['number_of_epochs'],
                    c['early_stopping_patience'],
                    c['grad_clip'],
                    c['lr_decay'],
                    scheduler)
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

    def pitch_rows(self) -> tuple[np.ndarray, int]:
        """For note tokens: an array with, for every token id, the row of its pitch among the distinct
        pitches of the vocabulary (the rest counts as a pitch), and the row of the rest (-1 if the
        vocabulary has no rest token). Used by the staged sampling in draw()."""
        if self._pitch_rows is None:
            names = [split_note_token(self.map_reverse[idx])[0] for idx in range(self.vocabulary_size)]
            rows = {name: row for row, name in enumerate(sorted(set(names)))}
            self._pitch_rows = (np.array([rows[name] for name in names]), rows.get(REST_TOKEN, -1))
        return self._pitch_rows

    def draw(self, logits: torch.Tensor) -> tuple[float, int]:
        """Draws the next token from logits of shape (1, vocabulary) with the sampling settings of
        config.ini; returns (top probability, token id).

        With rhythm_temperature > 0 the note model samples in stages, with separate temperatures for
        melody and rhythm (see sample_staged). Otherwise one temperature applies to whole tokens.
        """
        c = self.c
        if c['rhythm_temperature'] > 0 and self.kind == 'note':
            token_pitch, rest_row = self.pitch_rows()
            return sample_staged(logits, token_pitch, rest_row, c['temperature'], c['rhythm_temperature'], c['top_p'])
        return sample(logits, c['temperature'], c['top_p'])

    def next_token(self, seq: list[int], motifs: dict[str, int]) -> int:
        """Samples the next token id. When the model is unsure (top probability at or below
        motif_threshold) a matching corpus motif is continued instead, if there is one."""
        with torch.no_grad():
            # Only the prediction after the last token is needed.
            logits = self.model(torch.tensor([seq], dtype=torch.long))[:, -1, :]
        prob_max, token = self.draw(logits)
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


def _sharpen(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Applies a temperature to probabilities: p ** (1 / temperature), renormalised. Below 1 the likely
    choices become likelier, above 1 the distribution flattens; zero entries stay zero."""
    powered = np.power(probabilities, 1.0 / temperature)
    return powered / np.sum(powered)


def sample_staged(logits: torch.Tensor, token_pitch: np.ndarray, rest_row: int, temperature: float = 1.0,
                  rhythm_temperature: float = 1.0, top_p: float = 1.0) -> tuple[float, int]:
    """Samples a note token in three stages, so that melody and rhythm can have different temperatures.

    A single temperature on whole tokens cannot serve both: low enough to keep the melody in key, it
    makes the rhythm monotonous (the generated durations are far less varied than real ones); high
    enough for a lively rhythm, the melody wanders through keys. The stages use the model's own
    probabilities P(token), only split up by the chain rule:

    1. Rest or note? P(rest) is the summed probability of the rest tokens. This is a rhythmic decision,
       so it is sharpened with rhythm_temperature.
    2. Which pitch, if it is a note? P(pitch) is the summed probability of all tokens with that pitch,
       sharpened with `temperature`, and optionally cut to the nucleus top_p.
    3. How long? Among the tokens of the chosen pitch (or the rest tokens), P(token) sharpened with
       rhythm_temperature.

    logits: (1, vocabulary). token_pitch: for every token id the row of its pitch among the distinct
    pitches; rest_row: the row of the rest, or -1 without rest tokens (see Worker.pitch_rows).
    Returns (top probability, token id). The top probability is that of the most likely whole token at
    `temperature`, the same quantity sample() returns, so motif_threshold means the same in both.
    """
    scaled = logits.detach().numpy().astype(np.float64)[0]
    token_probability = np.exp(scaled - np.max(scaled))
    token_probability = token_probability / np.sum(token_probability)             # P(token) at temperature 1
    prob_max = float(np.max(_sharpen(token_probability, temperature)))
    pitch_probability = np.bincount(token_pitch, weights=token_probability)       # P(pitch), rest included

    # Stage 1: rest or note.
    is_rest = False
    if rest_row >= 0:
        rest = min(max(float(pitch_probability[rest_row]), 0.0), 1.0)
        rest_or_note = _sharpen(np.array([rest, 1.0 - rest]), rhythm_temperature)
        is_rest = bool(np.random.random() < rest_or_note[0])

    # Stage 2: the pitch.
    if is_rest:
        row = rest_row
    else:
        notes = pitch_probability.copy()
        if rest_row >= 0:
            notes[rest_row] = 0.0
        notes = _sharpen(notes / np.sum(notes), temperature)
        if top_p < 1.0:
            order = np.argsort(-notes)
            keep = order[:int(np.searchsorted(np.cumsum(notes[order]), top_p)) + 1]
            nucleus = np.zeros_like(notes)
            nucleus[keep] = notes[keep]
            notes = nucleus / np.sum(nucleus)
        row = int(np.random.choice(len(notes), p=notes))

    # Stage 3: the duration, among the tokens that have the chosen pitch.
    candidates = np.flatnonzero(token_pitch == row)
    durations = _sharpen(token_probability[candidates] / np.sum(token_probability[candidates]), rhythm_temperature)
    return prob_max, int(candidates[np.random.choice(len(candidates), p=durations)])


def make_scheduler(optimizer: AdamW, lr_schedule: str, warmup_steps: int, total_steps: int) -> Optional[LambdaLR]:
    """The learning-rate schedule named by config.ini's lr_schedule, stepped once per batch, or None.

    'constant': no scheduler (returns None). The learning rate stays at learning_rate, apart from
        train_model's drop on epochs without improvement (lr_decay).
    'cosine':   the learning rate rises linearly from near 0 to learning_rate over the first warmup_steps
        batches, then falls along half a cosine wave to 0 at the last batch of the last epoch:

            factor(step) = min(1, (step + 1) / warmup_steps) * 0.5 * (1 + cos(pi * step / total_steps))

        The warm-up keeps the first, badly scaled updates small; the slow fall lets the network take big
        steps early and settle into a good solution at the end. The run then has a fixed length.
    """
    if lr_schedule == 'constant':
        return None
    if lr_schedule != 'cosine':
        raise ValueError(f'Unknown lr_schedule "{lr_schedule}" in config.ini (use constant or cosine).')

    def factor(step: int) -> float:
        warmup = min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
        return warmup * 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / max(total_steps, 1))))

    return LambdaLR(optimizer, factor)


def train_model(model: Module,
                train_batches: Iterable,
                valid_loader: DataLoader,
                optimizer: AdamW,
                csv_logger: str,
                pth_best: str,
                num_epochs: int = 10,
                patience: int = 0,
                grad_clip: float = 0.0,
                lr_decay: float = 0.0,
                scheduler: Optional[LambdaLR] = None):
    """The training loop. Losses are per-token averages over all non-padding positions.

    train_batches yields the (inputs, targets) batches of one epoch each time it is iterated: a
    DataLoader over fixed windows, or a RandomWindows sampler.

    After each epoch the weights are saved to pth_best if the validation loss improved. Otherwise, with
    0 < lr_decay < 1 and no scheduler, the best weights are reloaded and the learning rate is multiplied
    by lr_decay. A scheduler (see make_scheduler) is stepped after every batch and owns the learning rate,
    so lr_decay is ignored with it. Training stops after `patience` epochs in a row without improvement
    (0 means never early). One line per epoch goes to csv_logger: epoch, train_loss, validation_loss, best.
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
            for inputs, labels in train_batches:
                optimizer.zero_grad()
                outputs = model(inputs)
                tokens = int((labels != IGNORE_INDEX).sum())
                loss = loss_function(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
                (loss / max(tokens, 1)).backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
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
                if scheduler is None and 0 < lr_decay < 1:
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

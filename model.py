import collections
import json
import os
import random
import sys

import numpy as np
import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import Config, log
from data import get_dir, generate_input, generate_output

cfg = Config()
motif_augmentation = True
motif_threshold = 0.10

seed = 0
random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictions = 1024
IGNORE_INDEX = -100


class TorchModule(Module):
    def __init__(self, vocabulary_size: int, map_direct: dict[str, int], map_reverse: dict[int, str], hidden_size: int,
                 lora_enable: bool = False, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                 lora_targets: str = "ih,hh,out,emb", add_causal_attn: bool = True, attn_heads: int = 4):
        super(TorchModule, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.map_direct = map_direct
        self.map_reverse = map_reverse
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=hidden_size)
        self.dropout = torch.nn.Dropout(p=0.25)
        self.lstm = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=vocabulary_size)

        self.emb_post = None
        self.post_attn = None

        if add_causal_attn:
            self.post_attn = CausalSelfAttention(hidden_size, n_heads=attn_heads)

        if lora_enable:
            self.apply_lora(lora_r, lora_alpha, lora_dropout, lora_targets)

        self.to(device)

    def apply_lora(self, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                   lora_targets: str = "ih,hh,out,emb"):
        # Wraps the current weights with LoRA adapters. Call it after loading a trained base model
        # when fine-tuning, since the wrapped base weights are frozen.
        from lora import inject_lora_into_lstm, inject_lora_into_linear
        targets = tuple(x.strip() for x in lora_targets.split(',') if x.strip())
        if "emb" in targets:
            self.emb_post = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            torch.nn.init.eye_(self.emb_post.weight)
            inject_lora_into_linear(self.emb_post, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        if ("ih" in targets) or ("hh" in targets):
            inject_lora_into_lstm(self.lstm, r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
                                  targets=tuple(x for x in ("ih", "hh") if x in targets))
        if "out" in targets:
            inject_lora_into_linear(self.linear, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        self.to(device)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        x = self.embedding(x)
        if self.emb_post is not None:
            x = self.emb_post(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        if self.post_attn is not None:
            x = self.post_attn(x)
        # Logits for every timestep: (batch, time, vocabulary). Position t predicts token t + 1.
        x = self.linear(x)
        return x


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = torch.nn.LayerNorm(d_model)
        self.register_buffer("_mask", None, persistent=False)

    def _causal_mask(self, T: int, device):
        if (self._mask is None) or (self._mask.size(0) != T):
            m = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
            self._mask = m
        return self._mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        mask = self._causal_mask(T, x.device)
        y, _ = self.mha(x, x, x, attn_mask=mask)
        return self.ln(x + y)


class TorchDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert len(x) == len(y)
        self.x = torch.from_numpy(x).long().to(device)
        self.y = torch.from_numpy(y).long().to(device)

    def __len__(self) -> int:
        return min(len(self.x), len(self.y))

    def __getitem__(self, idx: int) -> (torch.LongTensor, torch.LongTensor):
        return self.x[idx], self.y[idx]


class Worker:
    def __init__(self, composer: str, instruments: [str], kind: str):
        self.composer = composer
        self.instruments = instruments
        self.kind = kind

        self.crt_dir = get_dir(self.composer, self.instruments)

        self.data, self.vocabulary_size, self.map_direct, self.map_reverse = Worker.__load_data__(self.composer, self.instruments, self.kind)

        if not os.path.exists(os.path.join(self.crt_dir, self.kind + '_motifs.json')):
            self.motifs = Worker.__motif_query_all__(self.map_direct, self.map_reverse, os.path.join(self.crt_dir, self.kind + '_input.txt'), self.kind in ('pitch', 'note'))
            with open(os.path.join(self.crt_dir, self.kind + '_motifs.json'), 'wt') as file:
                json.dump(self.motifs, file, indent=4)
        with open(os.path.join(self.crt_dir, self.kind + '_motifs.json'), 'rt') as file:
            self.motifs = json.load(file)

        # Split by whole parts, not by windows, so overlapping windows never leak across the split.
        data_trn, data_val = Worker.__split_data__(self.data, 0.8)
        number_of_steps = cfg.config[kind]['number_of_steps']
        window_stride = cfg.config[kind]['window_stride'] or max(1, number_of_steps // 4)
        self.d_trn = TorchDataset(*Worker.__generate_xy__(data_trn, number_of_steps, window_stride))
        self.d_val = TorchDataset(*Worker.__generate_xy__(data_val, number_of_steps, window_stride))

        self.model = None

    def train(self):
        if not os.path.exists(os.path.join(self.crt_dir, self.kind + '_model.torch')):
            c = cfg.config[self.kind]
            model = self.__new_model__(lora_enable=False)
            if c['base_model']:
                log(f"Loading base model {c['base_model']}...")
                model.load_state_dict(torch.load(c['base_model'], map_location=device))
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
            loss_function = CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='sum')
            optimizer = Adam([p for p in model.parameters() if p.requires_grad])
            train_loader = DataLoader(dataset=self.d_trn, batch_size=cfg.config[self.kind]['batch_size'], shuffle=True)
            valid_loader = DataLoader(dataset=self.d_val, batch_size=cfg.config[self.kind]['batch_size'])
            csv_logger = os.path.join(self.crt_dir, self.kind + '_log.csv')
            pth_best = os.path.join(self.crt_dir, self.kind + '_model_best.torch')

            Worker.__train_model__(model,
                                   train_loader,
                                   valid_loader,
                                   loss_function,
                                   optimizer,
                                   csv_logger,
                                   pth_best,
                                   cfg.config[self.kind]['number_of_epochs'],
                                   cfg.config[self.kind]['early_stopping_patience'])
            # Only publish the final model once training has finished, so an interrupted run is retrained.
            os.replace(pth_best, os.path.join(self.crt_dir, self.kind + '_model.torch'))

    def test(self):
        if self.model is None:
            self.model = self.__new_model__(lora_enable=cfg.config[self.kind]['lora_enable'])
            self.model.load_state_dict(torch.load(os.path.join(self.crt_dir, self.kind + '_model.torch'), map_location=device))
            self.model = self.model.cpu()
        self.model.eval()

        number_of_steps = 0
        for key in cfg.config:
            number_of_steps = max(number_of_steps, cfg.config[key]['number_of_steps'])
        with open(os.path.join(self.crt_dir, self.kind + '_input.txt'), 'rt') as file:
            content = file.read().split()
            idx = random.randrange(0, len(content) - number_of_steps)
            inception = content[idx:idx + number_of_steps]
        sentence_ids = [self.map_direct[element] for element in inception]
        sentence = inception
        with torch.no_grad():
            for _ in range(predictions - number_of_steps):
                i = sentence_ids[-number_of_steps:]
                p, o = Worker.__motif_predict__(self.motifs, self.model, i, cfg.config[self.kind]['temperature'])
                sentence_ids.append(o)
                sentence.append(self.map_reverse[o])
        sentence = ' '.join(sentence)
        with open(os.path.join(self.crt_dir, self.kind + '_output.txt'), 'wt') as file:
            file.write(sentence)

    def __new_model__(self, lora_enable: bool) -> TorchModule:
        c = cfg.config[self.kind]
        return TorchModule(self.vocabulary_size, self.map_direct, self.map_reverse, c['hidden_size'],
                           lora_enable=lora_enable,
                           lora_r=c['lora_r'],
                           lora_alpha=c['lora_alpha'],
                           lora_dropout=c['lora_dropout'],
                           lora_targets=c['lora_targets'],
                           add_causal_attn=c['add_causal_attn'],
                           attn_heads=c['attn_heads'])

    @staticmethod
    def __read_sentences__(pth: str) -> [[str]]:
        data = []
        with open(pth, 'rt') as file:
            for sentence in file.read().strip('\n').split('\n'):
                data.append(sentence.split())
        return data

    @staticmethod
    def __build_vocabulary__(pth: str) -> dict[str, int]:
        data = []
        for sentence in Worker.__read_sentences__(pth):
            data += [word for word in sentence]
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        elements, _ = list(zip(*count_pairs))
        map_direct = dict(zip(elements, range(len(elements))))
        assert '' not in map_direct
        return map_direct

    @staticmethod
    def __file_to_idx__(file: str, map_direct: dict[str, int]) -> [[int]]:
        data = []
        for sentence in Worker.__read_sentences__(file):
            data.append([map_direct[word] for word in sentence])
        return data

    @staticmethod
    def __load_data__(composer: str, instruments: [str], kind: str) -> ([[int]], int, dict[str, int], dict[int, str]):
        crt_dir = get_dir(composer, instruments)
        pth = os.path.join(crt_dir, kind + '_input.txt')
        map_direct = Worker.__build_vocabulary__(pth)
        data = Worker.__file_to_idx__(pth, map_direct)
        vocabulary_size = len(map_direct)
        map_reverse = dict(zip(map_direct.values(), map_direct.keys()))
        return data, vocabulary_size, map_direct, map_reverse

    @staticmethod
    def __split_data__(data: [[int]], ratio: float) -> ([[int]], [[int]]):
        total = sum(len(sentence) for sentence in data)
        running = 0
        for idx, sentence in enumerate(data):
            if running >= total * ratio:
                return data[:idx], data[idx:]
            running += len(sentence)
        return data[:-1], data[-1:]

    @staticmethod
    def __generate_xy__(data: [[int]], number_of_steps: int, window_stride: int) -> (np.ndarray, np.ndarray):
        # Each sample is a window of number_of_steps tokens; its target is the same window shifted by one,
        # so the loss is computed at every timestep instead of only the last one.
        # Parts shorter than a window are right-padded: inputs with token 0, targets with IGNORE_INDEX.
        # The model is causal, so padding at the end never influences the real positions.
        x = []
        y = []
        for sentence in data:
            if len(sentence) < 2:
                continue
            last_start = max(0, len(sentence) - 1 - number_of_steps)
            starts = list(range(0, last_start + 1, window_stride))
            if starts[-1] != last_start:
                starts.append(last_start)
            for start in starts:
                window = sentence[start:start + number_of_steps + 1]
                pad = number_of_steps + 1 - len(window)
                x.append(window[:-1] + [0] * pad)
                y.append(window[1:] + [IGNORE_INDEX] * pad)
        x = np.array(x, dtype=int).reshape((-1, number_of_steps))
        y = np.array(y, dtype=int).reshape((-1, number_of_steps))
        assert len(x) == len(y)
        return x, y

    @staticmethod
    def __train_model__(model: Module,
                        train_loader: DataLoader,
                        valid_loader: DataLoader,
                        loss_function: CrossEntropyLoss,
                        optimizer: Adam,
                        csv_logger: str,
                        pth_best: str,
                        num_epochs: int = 10,
                        patience: int = 0):
        best_valid_loss = float('inf')
        epochs_without_improvement = 0
        with open(csv_logger, 'wt') as file:
            file.write('epoch,train_loss,validation_loss,best\n')
            for epoch in tqdm(range(num_epochs)):
                model.train()
                total_train_loss = 0.0
                total_train_tokens = 0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    tokens = int((labels != IGNORE_INDEX).sum())
                    loss = loss_function(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
                    (loss / max(tokens, 1)).backward()
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

                # Per-token averages, so padded positions and uneven batches do not skew the numbers.
                avg_train_loss = total_train_loss / max(total_train_tokens, 1)
                avg_valid_loss = total_valid_loss / max(total_valid_tokens, 1)
                improved = avg_valid_loss < best_valid_loss
                if improved:
                    best_valid_loss = avg_valid_loss
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), pth_best)
                else:
                    epochs_without_improvement += 1
                file.write(f'{epoch + 1},{avg_train_loss:.4f},{avg_valid_loss:.4f},{int(improved)}\n')
                file.flush()
                if 0 < patience <= epochs_without_improvement:
                    break

    @staticmethod
    def __temp_sample__(pred: torch.FloatTensor, temp: float = 1.0) -> (float, int):
        pred = pred.detach().numpy().astype(np.float64)[0]
        pred = np.exp(pred / temp)
        pred = pred / np.sum(pred)
        prob = np.random.multinomial(1, pred, 1)
        return np.max(pred), np.argmax(prob)

    @staticmethod
    def __temp_predict__(model: Module, seq: list[int], temp: float = 1.0) -> (float, int):
        with torch.no_grad():
            pred = model(torch.from_numpy(np.array(seq, dtype=int).reshape((1, -1))).long())[:, -1, :]
        return Worker.__temp_sample__(pred, temp)

    @staticmethod
    def __motif_predict__(motifs: dict[str, int], model: Module, seq: list[int], temp: float = 1.0) -> (float, int):
        prob_max, prob_argmax = Worker.__temp_predict__(model, seq, temp)
        if prob_max > motif_threshold or not motif_augmentation:
            return prob_max, prob_argmax

        motifs_str = list(motifs.keys())
        for motif in motifs_str:
            motif = [model.map_direct[word] for word in motif.split()]
            length = min(len(seq), len(motif) - 1)
            if seq[-length:] == motif[-length - 1:-1]:
                return 1.0, motif[-1]

        return 0.0, prob_argmax

    @staticmethod
    def __motif_query_filter__(motif_int: [int]) -> bool:
        return len(set(motif_int)) > 1

    @staticmethod
    def __motif_query_any__(map_direct: dict[str, int],
                            map_reverse: dict[int, str],
                            pth: str,
                            motif_length: int,
                            motif_filter: bool = False) -> dict[str, int]:
        elems_int = []
        with open(pth, 'rt') as file:
            for sentence in file.read().strip('\n').split('\n'):
                for word in sentence.split():
                    elems_int.append(map_direct[word])
        counts = collections.Counter(tuple(elems_int[motif_start:motif_start + motif_length])
                                     for motif_start in range(len(elems_int) - motif_length))
        motifs: dict[str, int] = {}
        for motif_int, motif_occ in counts.items():
            if motif_filter and not Worker.__motif_query_filter__(list(motif_int)):
                continue
            if motif_occ > 1:
                motifs[' '.join(map_reverse[word] for word in motif_int)] = motif_occ

        return dict(sorted(motifs.items(), key=lambda item: item[1], reverse=True))

    @staticmethod
    def __motif_query_all__(map_direct: dict[str, int],
                            map_reverse: dict[int, str],
                            pth: str,
                            motif_filter: bool = False) -> dict[str, int]:
        length_lower_bound = 4
        length_upper_bound = 8
        motifs = {}

        for motif_length in range(length_lower_bound, length_upper_bound + 1):
            motifs.update(Worker.__motif_query_any__(map_direct, map_reverse, pth, motif_length, motif_filter))

        return dict(sorted(motifs.items(), key=lambda item: item[1], reverse=True))


def main_train(composer: str, instruments: [str]):
    generate_input(composer, instruments)
    Worker(composer=composer, instruments=instruments, kind='note').train()


def main_test(composer: str, instruments: [str]):
    Worker(composer=composer, instruments=instruments, kind='note').test()
    generate_output(composer, instruments)


if __name__ == '__main__':
    main_train(sys.argv[1], sys.argv[2:])
    main_test(sys.argv[1], sys.argv[2:])

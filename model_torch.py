import collections
import multiprocessing
import os
import random
import sys

import numpy as np
import regex as re
import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import Config
from data import get_dir, generate_input, generate_output

cfg = Config().config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictions = 1000


class TorchModule(Module):
    def __init__(self, vocabulary_size: int, map_direct: dict[str, int], map_reverse: dict[int, str], hidden_size: int):
        super(TorchModule, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.map_direct = map_direct
        self.map_reverse = map_reverse
        self.embedding = torch.nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=hidden_size)
        self.dropout = torch.nn.Dropout(p=0.25)
        self.lstm = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=vocabulary_size)
        self.to(device)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x


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
    def __init__(self, composer: str, instruments: [str], kind: str, mode: str):
        crt_dir = get_dir(composer, instruments)

        data, vocabulary_size, map_direct, map_reverse = Worker.__load_data__(composer, instruments, kind)
        motifs = Worker.__motif_query_all__(map_direct, map_reverse, os.path.join(crt_dir, kind + '_input.txt'))
        print(kind, 'vocabulary size is', vocabulary_size)
        x, y = Worker.__generate_xy__(data, cfg[kind]['number_of_steps'])
        split = int(min(len(x), len(y)) * 0.8)
        d_trn = TorchDataset(x[:split], y[:split])
        d_val = TorchDataset(x[split:], y[split:])

        if mode == 'train' and not os.path.exists(os.path.join(crt_dir, kind + '_model.torch')):
            model = TorchModule(vocabulary_size, map_direct, map_reverse, cfg[kind]['hidden_size'])
            loss_function = CrossEntropyLoss()
            optimizer = Adam(model.parameters())
            train_loader = DataLoader(dataset=d_trn, batch_size=cfg[kind]['batch_size'], shuffle=True)
            valid_loader = DataLoader(dataset=d_val, batch_size=cfg[kind]['batch_size'])
            csv_logger = os.path.join(crt_dir, kind + '_log.csv')

            Worker.__train_model__(model,
                                   train_loader,
                                   valid_loader,
                                   loss_function,
                                   optimizer,
                                   csv_logger,
                                   cfg[kind]['number_of_epochs'])
            torch.save(model.state_dict(), os.path.join(crt_dir, kind + '_model.torch'))

        if mode == 'test' and not os.path.exists(os.path.join(crt_dir, kind + '_output.txt')):
            model = TorchModule(vocabulary_size, map_direct, map_reverse, cfg[kind]['hidden_size']).cpu()
            model.load_state_dict(torch.load(os.path.join(crt_dir, kind + '_model.torch')))

            number_of_steps = 0
            for key in cfg:
                number_of_steps = max(number_of_steps, cfg[key]['number_of_steps'])
            with open(os.path.join(crt_dir, kind + '_input.txt'), 'rt') as file:
                inception = file.read().split()[:number_of_steps]
            sentence_ids = [map_direct[element] for element in inception]
            sentence = inception
            for _ in tqdm(range(predictions)):
                i = sentence_ids[-number_of_steps:]
                o = Worker.__motif_predict__(motifs, model, i, cfg[kind]['temperature'])
                sentence_ids.append(o)
                sentence.append(map_reverse[o])
            sentence = ' '.join(sentence)
            with open(os.path.join(crt_dir, kind + '_output.txt'), 'wt') as file:
                file.write(sentence)

    @staticmethod
    def __read_sentences__(pth: str) -> [[str]]:
        data = []
        with open(pth, 'rt') as file:
            for sentence in file.read().split('\n'):
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
    def __generate_xy__(data: [[int]], number_of_steps: int) -> (np.ndarray, np.ndarray):
        x = []
        y = []
        for crt_idx_sentence in range(len(data)):
            for crt_idx_word in range(len(data[crt_idx_sentence]) - number_of_steps):
                x.append(data[crt_idx_sentence][crt_idx_word:crt_idx_word + number_of_steps])
                y.append(data[crt_idx_sentence][crt_idx_word + number_of_steps])
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        assert len(x) == len(y)
        return x, y

    @staticmethod
    def __train_model__(model: Module,
                        train_loader: DataLoader,
                        valid_loader: DataLoader,
                        loss_function: CrossEntropyLoss,
                        optimizer: Adam,
                        csv_logger: str,
                        num_epochs: int = 10):
        with open(csv_logger, 'wt') as log:
            log.write('epoch,train_loss,validation_loss\n')
            for epoch in tqdm(range(num_epochs)):
                model.train()
                total_train_loss = 0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()

                model.eval()
                total_valid_loss = 0
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        outputs = model(inputs)
                        loss = loss_function(outputs, labels)
                        total_valid_loss += loss.item()

                avg_train_loss = total_train_loss / len(train_loader)
                avg_valid_loss = total_valid_loss / len(valid_loader)
                log.write(f'{epoch + 1},{avg_train_loss:.4f},{avg_valid_loss:.4f}\n')

    @staticmethod
    def __temp_sample__(pred: torch.FloatTensor, temp: float = 1.0) -> int:
        pred = pred.detach().numpy().astype(np.float64)[0]
        pred = np.exp(pred / temp)
        prob = np.random.multinomial(1, pred / np.sum(pred), 1)
        return np.argmax(prob)

    @staticmethod
    def __temp_predict__(model: Module, seq: list[int], temp: float = 1.0) -> int:
        pred = model(torch.from_numpy(np.array(seq, dtype=int).reshape((1, -1))).long())
        return Worker.__temp_sample__(pred, temp)

    @staticmethod
    def __motif_predict__(motifs: dict[str, int], model: Module, seq: list[int], temp: float = 1.0) -> int:
        motifs_str = list(motifs.keys())
        random.shuffle(motifs_str)
        for motif in motifs_str:
            motif = [model.map_direct[word] for word in motif.split()]
            length = min(len(seq), len(motif) - 1)
            seq_sub = seq[-length:]
            motif_sub = motif[-length - 1:-1]
            assert len(seq_sub) == len(motif_sub) == length
            if seq_sub == motif_sub:
                return motif[-1]
        return Worker.__temp_predict__(model, seq, temp)

    @staticmethod
    def __motif_query_any__(map_direct: dict[str, int], map_reverse: dict[int, str], pth: str, motif_length: int) -> dict[str, int]:
        elems_int = []
        with open(pth, 'rt') as file:
            for sentence in file.read().split('\n'):
                for word in sentence.split():
                    elems_int.append(map_direct[word])
        elems_str = ' '.join(map_reverse[word] for word in elems_int)

        motifs: dict[str, int] = {}
        visited: set[str] = set()

        for motif_start in range(len(elems_int) - motif_length):
            motif_int: [int] = elems_int[motif_start:motif_start + motif_length]
            motif_str: str = ' '.join([map_reverse[word] for word in motif_int])
            if motif_str in visited:
                continue
            visited.add(motif_str)

            if motif_str not in motifs:
                motif_occ = len(re.findall(pattern=re.escape(pattern=motif_str,
                                                             special_only=False,
                                                             literal_spaces=False),
                                           string=elems_str,
                                           overlapped=True))
                if motif_occ > 1:
                    motifs[motif_str] = motif_occ

        return dict(sorted(motifs.items(), key=lambda item: item[1], reverse=True))

    @staticmethod
    def __motif_query_all__(map_direct: dict[str, int], map_reverse: dict[int, str], pth: str) -> dict[str, int]:
        length_lower_bound = 8
        length_upper_bound = 12
        motifs = {}

        with multiprocessing.Pool(multiprocessing.cpu_count() - 2) as pool:
            args = [(map_direct, map_reverse, pth, motif_length) for motif_length in range(length_lower_bound, length_upper_bound + 1)]
            for result in pool.starmap(Worker.__motif_query_any__, args):
                motifs.update(result)

        return dict(sorted(motifs.items(), key=lambda item: item[1], reverse=True))


def main(composer: str, instruments: [str]):
    generate_input(composer, instruments)
    for kind in cfg:
        Worker(composer=composer, instruments=instruments, kind=kind, mode='train')
    for kind in cfg:
        Worker(composer=composer, instruments=instruments, kind=kind, mode='test')
    generate_output(composer, instruments)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])

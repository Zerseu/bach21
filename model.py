import collections
import json
import multiprocessing
import os
import random
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import regex as re
import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import Config
from data import get_dir, generate_input
from entropy import sequence_entropy, composer_entropy

matplotlib.use('TkAgg')
cfg = Config()
motif_augmentation = True
motif_threshold = 0.25

seed = 0
random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    def __init__(self, composer: str, instruments: [str], kind: str):
        self.composer = composer
        self.instruments = instruments
        self.kind = kind

        self.crt_dir = get_dir(self.composer, self.instruments)

        self.data, self.vocabulary_size, self.map_direct, self.map_reverse = Worker.__load_data__(self.composer, self.instruments, self.kind)

        if not os.path.exists(os.path.join(self.crt_dir, self.kind + '_motifs.json')):
            self.motifs = Worker.__motif_query_all__(self.map_direct, self.map_reverse, os.path.join(self.crt_dir, self.kind + '_input.txt'), self.kind == 'pitch')
            with open(os.path.join(self.crt_dir, self.kind + '_motifs.json'), 'wt') as file:
                json.dump(self.motifs, file, indent=4)
        with open(os.path.join(self.crt_dir, self.kind + '_motifs.json'), 'rt') as file:
            self.motifs = json.load(file)

        x, y = Worker.__generate_xy__(self.data, cfg.config[kind]['number_of_steps'])
        split = int(min(len(x), len(y)) * 0.8)
        self.d_trn = TorchDataset(x[:split], y[:split])
        self.d_val = TorchDataset(x[split:], y[split:])

        self.model = None

    def train(self):
        if not os.path.exists(os.path.join(self.crt_dir, self.kind + '_model.torch')):
            model = TorchModule(self.vocabulary_size, self.map_direct, self.map_reverse, cfg.config[self.kind]['hidden_size'])
            loss_function = CrossEntropyLoss()
            optimizer = Adam(model.parameters())
            train_loader = DataLoader(dataset=self.d_trn, batch_size=cfg.config[self.kind]['batch_size'], shuffle=True)
            valid_loader = DataLoader(dataset=self.d_val, batch_size=cfg.config[self.kind]['batch_size'])
            csv_logger = os.path.join(self.crt_dir, self.kind + '_log.csv')

            Worker.__train_model__(model,
                                   train_loader,
                                   valid_loader,
                                   loss_function,
                                   optimizer,
                                   csv_logger,
                                   cfg.config[self.kind]['number_of_epochs'])
            torch.save(model.state_dict(), os.path.join(self.crt_dir, self.kind + '_model.torch'))

    def test(self):
        if self.model is None:
            self.model = TorchModule(self.vocabulary_size, self.map_direct, self.map_reverse, cfg.config[self.kind]['hidden_size']).cpu()
            self.model.load_state_dict(torch.load(os.path.join(self.crt_dir, self.kind + '_model.torch')))

        number_of_steps = 0
        for key in cfg.config:
            number_of_steps = max(number_of_steps, cfg.config[key]['number_of_steps'])
        with open(os.path.join(self.crt_dir, self.kind + '_input.txt'), 'rt') as file:
            content = file.read().split()
            idx = random.randrange(0, len(content) - number_of_steps)
            inception = content[idx:idx + number_of_steps]
        sentence_ids = [self.map_direct[element] for element in inception]
        sentence = inception
        for _ in range(predictions - number_of_steps):
            i = sentence_ids[-number_of_steps:]
            p, o = Worker.__motif_predict__(self.motifs, self.model, i, cfg.config[self.kind]['temperature'])
            sentence_ids.append(o)
            sentence.append(self.map_reverse[o])
        sentence = ' '.join(sentence)
        with open(os.path.join(self.crt_dir, self.kind + '_output.txt'), 'wt') as file:
            file.write(sentence)

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
        with open(csv_logger, 'wt') as file:
            file.write('epoch,train_loss,validation_loss\n')
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
                file.write(f'{epoch + 1},{avg_train_loss:.4f},{avg_valid_loss:.4f}\n')

    @staticmethod
    def __temp_sample__(pred: torch.FloatTensor, temp: float = 1.0) -> (float, int):
        pred = pred.detach().numpy().astype(np.float64)[0]
        pred = np.exp(pred / temp)
        pred = pred / np.sum(pred)
        prob = np.random.multinomial(1, pred, 1)
        return np.max(pred), np.argmax(prob)

    @staticmethod
    def __temp_predict__(model: Module, seq: list[int], temp: float = 1.0) -> (float, int):
        pred = model(torch.from_numpy(np.array(seq, dtype=int).reshape((1, -1))).long())
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
        elems_str = ' '.join(map_reverse[word] for word in elems_int)

        motifs: dict[str, int] = {}
        visited: set[str] = set()

        for motif_start in range(len(elems_int) - motif_length):
            motif_int: [int] = elems_int[motif_start:motif_start + motif_length]
            if motif_filter and not Worker.__motif_query_filter__(motif_int):
                continue
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
    def __motif_query_all__(map_direct: dict[str, int],
                            map_reverse: dict[int, str],
                            pth: str,
                            motif_filter: bool = False) -> dict[str, int]:
        length_lower_bound = 4
        length_upper_bound = 8
        motifs = {}

        with multiprocessing.Pool(multiprocessing.cpu_count() - 2) as pool:
            args = [(map_direct, map_reverse, pth, motif_length, motif_filter) for motif_length in range(length_lower_bound, length_upper_bound + 1)]
            for result in pool.starmap(Worker.__motif_query_any__, args):
                motifs.update(result)

        return dict(sorted(motifs.items(), key=lambda item: item[1], reverse=True))


def main_train(composer: str, instruments: [str]):
    generate_input(composer, instruments)
    Worker(composer=composer, instruments=instruments, kind='pitch').train()


def main_test(composer: str, instruments: [str]):
    global motif_augmentation, motif_threshold
    generate_input(composer, instruments)
    expected_entropy, noise_entropy = composer_entropy(composer, instruments)
    crt_dir = get_dir(composer, instruments)

    worker = Worker(composer=composer, instruments=instruments, kind='pitch')

    temp_min = 10
    temp_max = 100
    no_trials = 10
    thresholds = [0.10, 0.25]

    with open(os.path.join(crt_dir, 'results.csv'), 'wt') as report:
        report.write('Motif Threshold, Sampling Temperature, Expected Entropy, Noise Entropy')
        for trial in range(no_trials):
            report.write(f', Without Motifs Trial #{trial + 1}')
        for trial in range(no_trials):
            report.write(f', With Motifs Trial #{trial + 1}')
        report.write('\n')
        report.flush()

        for threshold in thresholds:
            motif_threshold = threshold

            xs = []
            ys_wo = []
            ys_w = []
            ys_ref = []
            ys_nz = []
            for temperature in tqdm(range(temp_min, temp_max, 2)):
                temperature /= 10
                xs.append(temperature)
                cfg.config['pitch']['temperature'] = temperature

                motif_augmentation = False
                without_motifs = []
                for _ in range(no_trials):
                    worker.test()
                    with open(os.path.join(crt_dir, 'pitch_output.txt'), 'rt') as file:
                        sentence = file.read().split()
                    assert len(sentence) == predictions
                    without_motifs.append(sequence_entropy(sentence))

                motif_augmentation = True
                with_motifs = []
                for _ in range(no_trials):
                    worker.test()
                    with open(os.path.join(crt_dir, 'pitch_output.txt'), 'rt') as file:
                        sentence = file.read().split()
                    assert len(sentence) == predictions
                    with_motifs.append(sequence_entropy(sentence))

                ys_wo.append(np.mean(without_motifs))
                ys_w.append(np.mean(with_motifs))
                ys_ref.append(expected_entropy)
                ys_nz.append(noise_entropy)

                report.write(f'{threshold}, {temperature}, {expected_entropy}, {noise_entropy}')
                for trial in range(no_trials):
                    report.write(f', {without_motifs[trial]}')
                for trial in range(no_trials):
                    report.write(f', {with_motifs[trial]}')
                report.write('\n')
                report.flush()

            pth_plot = os.path.join(crt_dir, f'entropy_plot_motifs_thr_{motif_threshold:.2f}.png')
            dpi = 72
            fig_width = 3000
            fig_height = 1000
            plt.figure(figsize=(fig_width / dpi, fig_height / dpi), dpi=dpi)
            plt.plot(xs, ys_wo, linestyle='-', color='red', label='Without Motifs')
            plt.plot(xs, ys_w, linestyle='-', color='green', label='With Motifs')
            plt.plot(xs, ys_ref, linestyle='--', color='orange', label='Expected Entropy')
            plt.plot(xs, ys_nz, linestyle='--', color='orange', label='Noise Entropy')
            plt.xlabel('Sampling Temperature')
            plt.ylabel(f'Avg. Entropy ({no_trials} Trials)')
            plt.title(composer.capitalize())
            plt.legend()
            plt.savefig(pth_plot, dpi=dpi, bbox_inches='tight')
            plt.close('all')


if __name__ == '__main__':
    main_train(sys.argv[1], sys.argv[2:])
    main_test(sys.argv[1], sys.argv[2:])

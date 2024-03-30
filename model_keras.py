import collections
import os
import sys

import numpy as np
from keras.callbacks import CSVLogger
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from tqdm import tqdm

from config import Config
from data import get_dir, generate_input, generate_output

cfg = Config().config
predictions = 1000


class Worker:
    def __init__(self, composer: str, instruments: [str], kind: str, mode: str):
        crt_dir = get_dir(composer, instruments)

        data, vocabulary_size, map_direct, map_reverse = Worker.__load_data__(composer, instruments, kind)
        print(kind, 'vocabulary size is', vocabulary_size)
        x, y = Worker.__generate_xy__(data, cfg[kind]['number_of_steps'], vocabulary_size)

        model = Sequential()
        model.add(Embedding(name='embedding', input_dim=vocabulary_size, output_dim=cfg[kind]['hidden_size']))
        model.add(SpatialDropout1D(name='dropout', rate=0.25))
        model.add(LSTM(name='lstm_1', units=cfg[kind]['hidden_size'], return_sequences=True))
        model.add(LSTM(name='lstm_2', units=cfg[kind]['hidden_size'], return_sequences=False))
        model.add(Dense(name='dense', units=vocabulary_size, activation='softmax'))
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=[CategoricalAccuracy()])

        logger = CSVLogger(filename=os.path.join(crt_dir, kind + '_log.csv'),
                           separator=',',
                           append=False)

        if mode == 'train' and not os.path.exists(os.path.join(crt_dir, kind + '_model.keras')):
            model.fit(x=x, y=y,
                      batch_size=cfg[kind]['batch_size'],
                      epochs=cfg[kind]['number_of_epochs'],
                      callbacks=[logger])

            model.save(os.path.join(crt_dir, kind + '_model.keras'))

        if mode == 'test' and not os.path.exists(os.path.join(crt_dir, kind + '_output.txt')):
            model = load_model(os.path.join(crt_dir, kind + '_model.keras'))

            number_of_steps = 0
            for key in cfg:
                number_of_steps = max(number_of_steps, cfg[key]['number_of_steps'])

            with open(os.path.join(crt_dir, kind + '_input.txt'), 'rt') as file:
                inception = file.read().split()[:number_of_steps]

            sentence_ids = [map_direct[element] for element in inception]
            sentence = inception
            for _ in tqdm(range(predictions)):
                i = np.array(sentence_ids[-number_of_steps:], dtype=int).reshape((1, number_of_steps))
                o = Worker.__temp_predict__(model, i, cfg[kind]['temperature'])
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
    def __build_vocabulary__(pth: str) -> dict:
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
    def __file_to_idx__(file: str, map_direct: dict) -> [[int]]:
        data = []
        for sentence in Worker.__read_sentences__(file):
            data.append([map_direct[word] for word in sentence])
        return data

    @staticmethod
    def __load_data__(composer: str, instruments: [str], kind: str) -> ([[int]], int, dict, dict):
        crt_dir = get_dir(composer, instruments)
        pth = os.path.join(crt_dir, kind + '_input.txt')
        map_direct = Worker.__build_vocabulary__(pth)
        data = Worker.__file_to_idx__(pth, map_direct)
        vocabulary_size = len(map_direct)
        map_reverse = dict(zip(map_direct.values(), map_direct.keys()))
        return data, vocabulary_size, map_direct, map_reverse

    @staticmethod
    def __generate_xy__(data: [[int]], number_of_steps: int, vocabulary_size: int) -> (np.ndarray, np.ndarray):
        x = []
        y = []
        for crt_idx_sentence in range(len(data)):
            for crt_idx_word in range(len(data[crt_idx_sentence]) - number_of_steps):
                x.append(data[crt_idx_sentence][crt_idx_word:crt_idx_word + number_of_steps])
                y.append(to_categorical(data[crt_idx_sentence][crt_idx_word + number_of_steps], vocabulary_size))
        x = np.array(x, dtype=int).reshape((-1, number_of_steps))
        y = np.array(y, dtype=int).reshape((-1, vocabulary_size))
        assert len(x) == len(y)
        return x, y

    @staticmethod
    def __temp_sample__(pred: np.ndarray, temp: float = 1.0) -> int:
        pred = np.asarray(pred).astype(np.float64)
        pred = np.log(pred + 1e-9) / temp
        pred = np.exp(pred)
        pred = pred / np.sum(pred)
        prob = np.random.multinomial(1, pred, 1)
        return np.argmax(prob)

    @staticmethod
    def __temp_predict__(model: Sequential, seq: np.ndarray, temp: float = 1.0):
        pred = model.predict(seq, verbose=0)[0]
        return Worker.__temp_sample__(pred, temp)


def main(composer: str, instruments: [str]):
    generate_input(composer, instruments)
    for kind in cfg:
        Worker(composer=composer, instruments=instruments, kind=kind, mode='train')
    for kind in cfg:
        Worker(composer=composer, instruments=instruments, kind=kind, mode='test')
    generate_output(composer, instruments)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])

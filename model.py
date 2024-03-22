import collections
import os
import random
import sys

import numpy as np
import tensorflow as tf
from keras.callbacks import CSVLogger
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

from config import Config
from data import get_dir, generate_input, generate_output

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

cfg = Config().config
predictions = 1000


class Model:
    def __init__(self, composer: str, instruments: [str], kind: str, mode: str):
        crt_dir = get_dir(composer, instruments)

        data, vocabulary_size, map_direct, map_reverse = Model.__load_data__(composer, instruments, kind)
        x, y = Model.__generate_xy__(data, cfg[kind]['number_of_steps'], vocabulary_size)

        model = Sequential()
        model.add(Embedding(name='embedding', input_dim=vocabulary_size, output_dim=cfg[kind]['hidden_size']))
        model.add(SpatialDropout1D(name='dropout', rate=0.2))
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
                      validation_split=0.2,
                      callbacks=[logger])

            model.save(os.path.join(crt_dir, kind + '_model.keras'))

        if mode == 'test' and not os.path.exists(os.path.join(crt_dir, kind + '_output.txt')):
            model = load_model(os.path.join(crt_dir, kind + '_model.keras'))

            number_of_steps = 0
            for key in cfg:
                number_of_steps = max(number_of_steps, cfg[key]['number_of_steps'])

            with open(os.path.join(crt_dir, kind + '_training.txt'), 'rt') as file:
                inception = file.read().split(' ')[:number_of_steps]

            sentence_ids = [map_direct[element] for element in inception]
            sentence = inception
            for n in range(predictions):
                if n % 100 == 0:
                    print('Prediction: ' + str(int(n / predictions * 100)) + '%')

                i = np.zeros((1, cfg[kind]['number_of_steps']))
                i[0] = np.array(sentence_ids[-cfg[kind]['number_of_steps']:])
                prediction = model.predict(i)
                o = np.argsort(prediction[:, cfg[kind]['number_of_steps'] - 1, :]).flatten()[::-1]
                w = o[0]

                sentence_ids.append(w)
                sentence.append(map_reverse[w])
            sentence = ' '.join(sentence)
            with open(os.path.join(crt_dir, kind + '_output.txt'), 'wt') as file:
                file.write(sentence)

    @staticmethod
    def __read_sentences__(pth: str) -> [[str]]:
        data = []
        with open(pth, 'rt') as file:
            for sentence in file.read().split('\n'):
                data.append(sentence.split(' '))
        return data

    @staticmethod
    def __build_vocabulary__(pth: str) -> dict:
        data = []
        for sentence in Model.__read_sentences__(pth):
            data += [word for word in sentence]
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        elements, _ = list(zip(*count_pairs))
        map_direct = dict(zip(elements, range(len(elements))))
        return map_direct

    @staticmethod
    def __file_to_ids__(file: str, map_direct: dict) -> [[int]]:
        data = []
        for sentence in Model.__read_sentences__(file):
            data.append([map_direct[word] for word in sentence])
        return data

    @staticmethod
    def __load_data__(composer: str, instruments: [str], kind: str) -> ([[int]], int, dict, dict):
        crt_dir = get_dir(composer, instruments)
        pth = os.path.join(crt_dir, kind + '_nlp.txt')
        map_direct = Model.__build_vocabulary__(pth)
        data = Model.__file_to_ids__(pth, map_direct)
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


def main(composer: str, instruments: [str]):
    generate_input(composer, instruments)
    for kind in cfg:
        Model(composer=composer, instruments=instruments, kind=kind, mode='train')
    for kind in cfg:
        Model(composer=composer, instruments=instruments, kind=kind, mode='test')
    generate_output(composer, instruments)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])

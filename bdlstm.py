import collections
import os
import random
import sys

import numpy as np
import tensorflow as tf
from keras import Input
from keras.callbacks import CSVLogger
from keras.layers import Activation, Dense, Dropout, Embedding, TimeDistributed, LSTM
from keras.models import load_model, Sequential
from keras.src.losses import CategoricalCrossentropy
from keras.src.metrics import CategoricalAccuracy
from keras.src.optimizers import Adam
from keras.utils import to_categorical

from config import Config
from data import get_dir, generate_input, generate_output

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

cfg = Config().config
predictions = 1000


class BDLSTMGenerator(object):
    def __init__(self, data: [int], number_of_steps: int, batch_size: int, vocabulary_size: int, skip_step: int):
        self.data = data
        self.number_of_steps = number_of_steps
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.current_index = 0
        self.skip_step = skip_step

    def generate(self) -> (np.ndarray, np.ndarray):
        x = np.zeros((self.batch_size, self.number_of_steps))
        y = np.zeros((self.batch_size, self.number_of_steps, self.vocabulary_size))
        while True:
            for i in range(self.batch_size):
                if self.current_index + self.number_of_steps >= len(self.data):
                    self.current_index = 0
                x[i, :] = self.data[self.current_index:self.current_index + self.number_of_steps]
                temp = self.data[self.current_index + 1:self.current_index + self.number_of_steps + 1]
                y[i, :, :] = to_categorical(temp, num_classes=self.vocabulary_size)
                self.current_index += self.skip_step
            yield x, y


class BDLSTM:
    def __init__(self, composer: str, instruments: [str], kind: str, mode: str):
        crt_dir = get_dir(composer, instruments)

        training_data, validation_data, vocabulary_size, map_direct, map_reverse = BDLSTM.__load_data__(composer, instruments, kind)

        training_data_generator = BDLSTMGenerator(training_data,
                                                  cfg[kind]['number_of_steps'],
                                                  cfg[kind]['batch_size'],
                                                  vocabulary_size,
                                                  skip_step=cfg[kind]['number_of_steps'])
        validation_data_generator = BDLSTMGenerator(validation_data,
                                                    cfg[kind]['number_of_steps'],
                                                    cfg[kind]['batch_size'],
                                                    vocabulary_size,
                                                    skip_step=cfg[kind]['number_of_steps'])

        model = Sequential()
        model.add(Input(name='input', shape=(cfg[kind]['number_of_steps'],)))
        model.add(Embedding(name='embedding', input_dim=vocabulary_size, output_dim=cfg[kind]['hidden_size']))
        model.add(LSTM(name='lstm', units=cfg[kind]['hidden_size'], return_sequences=True))
        model.add(Dropout(name='dropout', rate=0.25))
        model.add(TimeDistributed(name='time_dist_dense', layer=Dense(name='dense', units=vocabulary_size)))
        model.add(Activation(name='activation', activation='softmax'))
        model.compile(optimizer=Adam(),
                      loss=CategoricalCrossentropy(),
                      metrics=[CategoricalAccuracy()])

        logger = CSVLogger(filename=os.path.join(crt_dir, kind + '_log.csv'),
                           separator=',',
                           append=False)

        if mode == 'train' and not os.path.exists(os.path.join(crt_dir, kind + '_model.keras')):
            trn_steps = len(training_data) // (cfg[kind]['batch_size'] * cfg[kind]['number_of_steps'])
            val_steps = len(validation_data) // (cfg[kind]['batch_size'] * cfg[kind]['number_of_steps'])

            model.fit(training_data_generator.generate(),
                      steps_per_epoch=trn_steps,
                      epochs=cfg[kind]['number_of_epochs'],
                      validation_data=validation_data_generator.generate(),
                      validation_steps=val_steps,
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

                eps = 1E-3
                rnd = BDLSTM.__clamp_01__(random.random() + eps)
                base = cfg[kind]['temperature']
                idx = 0
                while not (1.0 / pow(base, idx + 1) < rnd <= 1.0 / pow(base, idx)) and idx < vocabulary_size - 1:
                    idx += 1
                w = o[idx]

                sentence_ids.append(w)
                sentence.append(map_reverse[w])

            sentence = ' '.join(sentence)
            with open(os.path.join(crt_dir, kind + '_output.txt'), 'wt') as file:
                file.write(sentence)

    @staticmethod
    def __clamp_01__(x: float) -> float:
        if x < 0:
            return 0
        if x > 1:
            return 1
        return x

    @staticmethod
    def __gen_eps__() -> float:
        return random.random() * 1E-3

    @staticmethod
    def __read_words__(training_path: str = None, validation_path: str = None) -> [str]:
        all_words = []
        if training_path is not None:
            with open(training_path, 'rt') as file:
                all_words += file.read().split(' ')
        if validation_path is not None:
            with open(validation_path, 'rt') as file:
                all_words += file.read().split(' ')
        return all_words

    @staticmethod
    def __build_vocabulary__(training_path: str, validation_path: str) -> dict:
        data = BDLSTM.__read_words__(training_path, validation_path)
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        elements, _ = list(zip(*count_pairs))
        map_direct = dict(zip(elements, range(len(elements))))
        return map_direct

    @staticmethod
    def __file_to_ids__(file: str, map_direct: dict) -> [int]:
        data = BDLSTM.__read_words__(file)
        return [map_direct[element] for element in data]

    @staticmethod
    def __load_data__(composer: str, instruments: [str], kind: str) -> ([int], [int], int, dict, dict):
        crt_dir = get_dir(composer, instruments)
        training_path = os.path.join(crt_dir, kind + '_training.txt')
        validation_path = os.path.join(crt_dir, kind + '_validation.txt')
        map_direct = BDLSTM.__build_vocabulary__(training_path, validation_path)
        training_data = BDLSTM.__file_to_ids__(training_path, map_direct)
        validation_data = BDLSTM.__file_to_ids__(validation_path, map_direct)
        vocabulary_size = len(map_direct)
        map_reverse = dict(zip(map_direct.values(), map_direct.keys()))
        return training_data, validation_data, vocabulary_size, map_direct, map_reverse


def main(composer: str, instruments: [str]):
    generate_input(composer, instruments)
    for kind in cfg:
        BDLSTM(composer=composer, instruments=instruments, kind=kind, mode='train')
    for kind in cfg:
        BDLSTM(composer=composer, instruments=instruments, kind=kind, mode='test')
    generate_output(composer, instruments)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])

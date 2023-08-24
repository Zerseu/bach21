import collections
import os
import random

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Activation, Dense, Dropout, Embedding, TimeDistributed, Bidirectional, LSTM, GRU
from keras.models import load_model, Sequential
from keras.utils import to_categorical

from config import Config
from data import generate_input

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class Constants:
    kinds = ['pitch', 'duration']
    root = 'data'


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
    _cfg = Config().config
    _predictions = 500

    def __init__(self, kind: str, mode: str):
        kind = Constants.kinds.index(kind)

        training_data, validation_data, vocabulary_size, map_direct, map_reverse = BDLSTM._load_data(Constants.kinds[kind])

        training_data_generator = BDLSTMGenerator(training_data,
                                                  BDLSTM._cfg[kind]['number_of_steps'],
                                                  BDLSTM._cfg[kind]['batch_size'],
                                                  vocabulary_size,
                                                  skip_step=BDLSTM._cfg[kind]['number_of_steps'])
        validation_data_generator = BDLSTMGenerator(validation_data,
                                                    BDLSTM._cfg[kind]['number_of_steps'],
                                                    BDLSTM._cfg[kind]['batch_size'],
                                                    vocabulary_size,
                                                    skip_step=BDLSTM._cfg[kind]['number_of_steps'])

        model = Sequential()
        model.add(Embedding(input_dim=vocabulary_size,
                            output_dim=BDLSTM._cfg[kind]['hidden_size'],
                            input_length=BDLSTM._cfg[kind]['number_of_steps']))
        model.add(GRU(units=BDLSTM._cfg[kind]['hidden_size'], return_sequences=True))
        model.add(Bidirectional(LSTM(units=BDLSTM._cfg[kind]['hidden_size'], return_sequences=True)))
        model.add(GRU(units=BDLSTM._cfg[kind]['hidden_size'], return_sequences=True))
        model.add(Dropout(rate=0.5))
        model.add(TimeDistributed(Dense(units=vocabulary_size)))
        model.add(Activation(activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        checkpoint = ModelCheckpoint(filepath=os.path.join(Constants.root, Constants.kinds[kind] + '_{epoch:03d}.hdf5'),
                                     monitor='val_loss',
                                     verbose=1,
                                     save_freq=int(1E9))
        logger = CSVLogger(filename=os.path.join(Constants.root, Constants.kinds[kind] + '_log.csv'),
                           separator=',',
                           append=False)

        if mode == 'train':
            trn_steps = len(training_data) // (BDLSTM._cfg[kind]['batch_size'] * BDLSTM._cfg[kind]['number_of_steps'])
            val_steps = len(validation_data) // (BDLSTM._cfg[kind]['batch_size'] * BDLSTM._cfg[kind]['number_of_steps'])

            model.fit(training_data_generator.generate(),
                      steps_per_epoch=trn_steps,
                      epochs=BDLSTM._cfg[kind]['number_of_epochs'],
                      validation_data=validation_data_generator.generate(),
                      validation_steps=val_steps,
                      callbacks=[checkpoint, logger])

            model.save(os.path.join(Constants.root, Constants.kinds[kind] + '_model.hdf5'))

        if mode == 'test':
            model = load_model(os.path.join(Constants.root, Constants.kinds[kind] + '_model.hdf5'))

            number_of_steps = 16
            inception = np.load(os.path.join(Constants.root, Constants.kinds[kind] + '_training.npy')).tolist()[
                        :number_of_steps]

            sentence_ids = [map_direct[element] for element in inception]
            sentence = inception
            for n in range(BDLSTM._predictions):
                if n % 100 == 0:
                    print('Prediction: ' + str(int(n / BDLSTM._predictions * 100)) + '%')

                i = np.zeros((1, BDLSTM._cfg[kind]['number_of_steps']))
                i[0] = np.array(sentence_ids[-BDLSTM._cfg[kind]['number_of_steps']:])
                prediction = model.predict(i)
                o = np.argsort(prediction[:, BDLSTM._cfg[kind]['number_of_steps'] - 1, :]).flatten()[::-1]

                eps = 1E-3
                rnd = BDLSTM._clamp_01(random.random() + eps)
                base = BDLSTM._cfg[kind]['temperature']
                idx = 0
                while not (1.0 / pow(base, idx + 1) < rnd <= 1.0 / pow(base, idx)) and idx < vocabulary_size - 1:
                    idx += 1
                w = o[idx]

                sentence_ids.append(w)
                sentence.append(map_reverse[w])

            sentence = np.array(sentence, dtype=int)
            np.save(os.path.join(Constants.root, Constants.kinds[kind] + '_output.npy'), sentence)

    @staticmethod
    def _clamp_01(x: float) -> float:
        if x < 0:
            return 0
        if x > 1:
            return 1
        return x

    @staticmethod
    def _gen_eps() -> float:
        return random.random() * 1E-3

    @staticmethod
    def _read_words(training_path: str = None, validation_path: str = None) -> [str]:
        all_words = []
        if training_path is not None:
            with open(training_path, 'rt') as file:
                all_words += file.read().split(' ')
        if validation_path is not None:
            with open(validation_path, 'rt') as file:
                all_words += file.read().split(' ')
        return all_words

    @staticmethod
    def _build_vocabulary(training_path: str, validation_path: str) -> dict:
        data = BDLSTM._read_words(training_path, validation_path)
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        elements, _ = list(zip(*count_pairs))
        map_direct = dict(zip(elements, range(len(elements))))
        return map_direct

    @staticmethod
    def _file_to_ids(file: str, map_direct: dict) -> [int]:
        data = BDLSTM._read_words(file)
        return [map_direct[element] for element in data]

    @staticmethod
    def _load_data(kind: str) -> ([int], [int], int, dict, dict):
        training_path = os.path.join(Constants.root, kind + '_training.txt')
        validation_path = os.path.join(Constants.root, kind + '_validation.txt')
        map_direct = BDLSTM._build_vocabulary(training_path, validation_path)
        training_data = BDLSTM._file_to_ids(training_path, map_direct)
        validation_data = BDLSTM._file_to_ids(validation_path, map_direct)
        vocabulary_size = len(map_direct)
        map_reverse = dict(zip(map_direct.values(), map_direct.keys()))
        return training_data, validation_data, vocabulary_size, map_direct, map_reverse


def main_bdlstm(train: bool = True, test: bool = False):
    generate_input()
    if train:
        for kind in Constants.kinds:
            if not os.path.exists(os.path.join(Constants.root, kind + '_model.hdf5')):
                BDLSTM(kind=kind, mode='train')
    if test:
        for kind in Constants.kinds:
            BDLSTM(kind=kind, mode='test')


if __name__ == '__main__':
    main_bdlstm()

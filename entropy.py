import os
import random
import sys

import numpy as np
import pandas as pd

from config import fprintf
from data import get_dir, generate_input


def helper_entropy(probabilities: pd.DataFrame) -> float:
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))


def reference_entropy(alphabet_size: int, sequence_length: int, trials=10) -> float:
    alphabet = [str(word) for word in range(alphabet_size)]
    trial_list = np.zeros(trials, dtype=float)
    for trial in range(trials):
        trial_list[trial] = sequence_entropy([random.choice(alphabet) for _ in range(sequence_length)])
    return np.mean(trial_list)


def sequence_entropy(sequence: [str]) -> float:
    df = pd.DataFrame({'current': sequence[:-1], 'next': sequence[1:]})
    transition_matrix = pd.crosstab(df['current'], df['next'], normalize='index')
    state_entropies = transition_matrix.apply(helper_entropy, axis=1)
    overall_entropy = state_entropies.mean()
    return overall_entropy


def composer_entropy(composer: str, instruments: [str]):
    crt_dir = get_dir(composer, instruments)
    generate_input(composer, instruments)
    pitches = []
    vocabulary = set()
    with open(os.path.join(crt_dir, 'pitch_input.txt'), 'rt') as file:
        for sentence in file.read().strip('\n').split('\n'):
            pitches.append([])
            for word in sentence.split():
                if word == 'RST':
                    continue
                else:
                    pitches[-1].append(word)
                    if word not in vocabulary:
                        vocabulary.add(word)
    for sequence in pitches:
        if len(sequence) > 256:
            fprintf(sequence_entropy(sequence), reference_entropy(len(vocabulary), len(sequence)))


def main(composer: str, instruments: [str]):
    composer_entropy(composer, instruments)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

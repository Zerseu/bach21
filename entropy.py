"""Entropy analysis of a composer's pitch sequences (the measurements behind the ESANN paper).

    python entropy.py bach violin

For every part longer than 256 notes it measures the transition entropy of the pitch sequence (the
average entropy of "which pitch comes next" per pitch) and compares it with random sequences over the
same alphabet. It saves two plots to bach21data/<composer>/<instrument>/ and prints the expected entropy
of a 1000-note piece next to the noise entropy at that length.

If that folder also has a results.csv (one row per motif threshold and sampling temperature, with 10
entropies measured without motifs and 10 with), it also plots entropy against temperature for each
threshold. Nothing in this repository writes results.csv; it comes from the paper's experiment runs.
"""
import os
import random
import sys

import matplotlib

matplotlib.use('Agg')  # write image files only; no window, works without a display

import matplotlib.pyplot as plt  # noqa: E402 (the backend must be chosen first)
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

from config import log
from data import REST_TOKEN, generate_input, get_dir

NOISE_TRIALS = 10


def helper_entropy(probabilities: pd.Series) -> float:
    """Shannon entropy in bits of one row of transition probabilities."""
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))


def reference_entropy(alphabet_size: int, sequence_length: int, trials: int = NOISE_TRIALS) -> float:
    """Average transition entropy of uniformly random sequences of the given length and alphabet size."""
    alphabet = [str(word) for word in range(alphabet_size)]
    trial_list = np.zeros(trials, dtype=float)
    for trial in range(trials):
        trial_list[trial] = sequence_entropy([random.choice(alphabet) for _ in range(sequence_length)])
    return np.mean(trial_list)


def sequence_entropy(sequence: list[str]) -> float:
    """Mean over states of the entropy of the next-symbol distribution."""
    df = pd.DataFrame({'current': sequence[:-1], 'next': sequence[1:]})
    transition_matrix = pd.crosstab(df['current'], df['next'], normalize='index')
    state_entropies = transition_matrix.apply(helper_entropy, axis=1)
    overall_entropy = state_entropies.mean()
    return overall_entropy


def composer_entropy(composer: str, instruments: list[str]) -> tuple[float, float]:
    """Plots the per-part entropies and returns (expected entropy at 1000 notes, noise entropy at 1000)."""
    log('Computing', composer.upper(), 'entropy plot...')
    crt_dir = get_dir(composer, instruments)
    generate_input(composer, instruments)
    pitches = []
    vocabulary = set()
    with open(os.path.join(crt_dir, 'pitch_input.txt'), 'rt') as file:
        for sentence in file.read().strip('\n').split('\n'):
            pitches.append([])
            for word in sentence.split():
                if word != REST_TOKEN:
                    pitches[-1].append(word)
                    vocabulary.add(word)

    seq_len_int = []
    seq_len_str = []
    x_values = []
    y_values = []
    for sequence in tqdm(pitches):
        if len(sequence) > 256:
            seq_len_int.append(len(sequence))
            seq_len_str.append(str(len(sequence)))
            x_values.append(sequence_entropy(sequence))
            y_values.append(reference_entropy(len(vocabulary), len(sequence)))
    seq_len_int, seq_len_str, x_values, y_values = zip(*sorted(zip(seq_len_int, seq_len_str, x_values, y_values)))
    seq_len_int = list(seq_len_int)
    seq_len_str = list(seq_len_str)
    x_values = list(x_values)
    y_values = list(y_values)

    dpi = 72
    fig_width = 1000
    fig_height = 500
    ln_width = 2

    pth_plot = os.path.join(crt_dir, 'entropy_plot_measured.png')
    plt.figure(figsize=(fig_width / dpi, fig_height / dpi), dpi=dpi)
    plt.bar(seq_len_str, x_values, label='Real Entropy')
    plt.bar(seq_len_str, y_values, bottom=x_values, label='Noise Entropy')
    plt.xlabel('Sequences')
    plt.ylabel('Entropy')
    plt.title(composer.capitalize())
    plt.legend()
    plt.savefig(pth_plot, dpi=dpi, bbox_inches='tight')
    plt.close('all')

    pth_plot = os.path.join(crt_dir, 'entropy_plot_interpolated.png')
    plt.figure(figsize=(fig_width / dpi, fig_height / dpi), dpi=dpi)
    interp = interp1d(seq_len_int, x_values, kind='linear')
    x_interp = np.linspace(min(seq_len_int), max(seq_len_int), 500).tolist()
    y_interp = [interp(x) for x in x_interp]
    plt.plot(x_interp, y_interp, linestyle='-', linewidth=ln_width, color='blue', label='Interpolated Entropy')
    plt.axvline(x=1000, linestyle='--', linewidth=ln_width, color='black', label='Length 1000')
    plt.xlabel('Sequence Length')
    plt.ylabel('Expected Entropy')
    plt.title(composer.capitalize())
    plt.legend()
    plt.savefig(pth_plot, dpi=dpi, bbox_inches='tight')
    plt.close('all')

    return interp(1000), reference_entropy(len(vocabulary), 1000)


def plot_motif_results(composer: str, crt_dir: str, thresholds: tuple[float, ...] = (0.10, 0.25)):
    """Entropy against sampling temperature, with and without motifs, from results.csv."""
    for threshold in thresholds:
        with open(os.path.join(crt_dir, 'results.csv'), 'rt') as report:
            report.readline()
            xs = []
            ys_wo = []
            ys_w = []
            ys_ref = []
            ys_nz = []
            for line in report.readlines():
                line = line.rstrip('\n')
                tokens = line.split(',')
                motif_threshold = float(tokens[0].strip(' '))
                sampling_temperature = float(tokens[1].strip(' '))
                expected_entropy = float(tokens[2].strip(' '))
                noise_entropy = float(tokens[3].strip(' '))
                without_motifs = [float(token.strip(' ')) for token in tokens[4:14]]
                with_motifs = [float(token.strip(' ')) for token in tokens[14:24]]

                if motif_threshold == threshold:
                    xs.append(sampling_temperature)
                    ys_wo.append(np.mean(without_motifs))
                    ys_w.append(np.mean(with_motifs))
                    ys_ref.append(expected_entropy)
                    ys_nz.append(noise_entropy)

            pth_plot = os.path.join(crt_dir, f'entropy_plot_motifs_thr_{threshold:.2f}.png')
            dpi = 72
            fig_width = 768
            fig_height = 512
            ln_width = 3
            plt.figure(figsize=(fig_width / dpi, fig_height / dpi), dpi=dpi)
            plt.plot(xs, ys_wo, linestyle='-', linewidth=ln_width, color='red', label='Without Motifs')
            plt.plot(xs, ys_w, linestyle='-', linewidth=ln_width, color='green', label='With Motifs')
            plt.plot(xs, ys_ref, linestyle='--', linewidth=ln_width, color='orange', label='Expected Entropy')
            plt.xlabel('Sampling Temperature')
            plt.ylabel(f'Avg. Entropy ({NOISE_TRIALS} Trials)')
            plt.title(composer.capitalize())
            plt.legend()
            plt.savefig(pth_plot, dpi=dpi, bbox_inches='tight')
            plt.close('all')


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python entropy.py COMPOSER [INSTRUMENT ...]')
    composer, instruments = sys.argv[1], sys.argv[2:]
    expected, noise = composer_entropy(composer, instruments)
    log(f'Expected entropy at 1000 notes: {float(expected):.4f} bits, noise entropy: {noise:.4f} bits')
    crt_dir = get_dir(composer, instruments)
    if os.path.exists(os.path.join(crt_dir, 'results.csv')):
        plot_motif_results(composer, crt_dir)
    else:
        log(f'No results.csv in {crt_dir}, skipping the motif plots.')


if __name__ == "__main__":
    main()

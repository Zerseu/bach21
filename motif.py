import os.path

import numpy as np
import stumpy
from music21 import pitch

from data import generate_input


def main():
    generate_input()

    pitches = []
    with open('data/pitch_training.txt', 'rt') as file:
        pitches += file.read().split(' ')
    with open('data/pitch_validation.txt', 'rt') as file:
        pitches += file.read().split(' ')

    sig = []
    for p in pitches:
        if p != 'RST':
            sig.append(pitch.Pitch(p).ps)
    sig = np.array(sig, dtype=float)

    window_size = 10
    if not os.path.exists('data/stump_{0}.npy'.format(window_size)):
        sig_profile = stumpy.stump(T_A=sig,
                                   m=window_size)
        np.save('data/stump_{0}'.format(window_size), sig_profile, allow_pickle=True)
    sig_profile = np.load('data/stump_{0}.npy'.format(window_size), allow_pickle=True)
    sig_motifs = stumpy.motifs(T=sig,
                               P=sig_profile[:, 0],
                               min_neighbors=3,
                               max_distance=0.0,
                               cutoff=0.0,
                               max_matches=1000,
                               max_motifs=100)
    print(sig_motifs)


if __name__ == "__main__":
    main()

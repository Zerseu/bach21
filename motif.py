import os.path

import numpy as np
import stumpy
from music21 import pitch

from data import generate_input


def main(motif_length: int = 10):
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

    if not os.path.exists('data/stump_{0}.npy'.format(motif_length)):
        sig_profile = stumpy.stump(T_A=sig,
                                   m=motif_length)
        np.save('data/stump_{0}'.format(motif_length), sig_profile, allow_pickle=True)
    sig_profile = np.load('data/stump_{0}.npy'.format(motif_length), allow_pickle=True)
    sig_motifs = stumpy.motifs(T=sig,
                               P=sig_profile[:, 0],
                               min_neighbors=9,
                               max_distance=0.0,
                               cutoff=0.0,
                               max_matches=1000,
                               max_motifs=1000)

    for m in sig_motifs[1]:
        motif_start = m[0]
        motif_ps = sig[motif_start:motif_start + motif_length]
        motif_ps = [pitch.Pitch(p) for p in motif_ps]
        motif_ps = [p.nameWithOctave for p in motif_ps]
        motif_occ = np.count_nonzero(m != -1)
        print(motif_ps, motif_occ)


if __name__ == "__main__":
    main()

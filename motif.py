import os.path

import graphviz
import numpy as np
import stumpy
from music21 import pitch

from data import generate_input


def query_any(motif_length: int = 10) -> [str]:
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
                               min_neighbors=5,
                               max_distance=0.0,
                               cutoff=0.0,
                               max_matches=10,
                               max_motifs=10000)

    motifs = []
    for m in sig_motifs[1]:
        motif_start = m[0]
        motif_ps = sig[motif_start:motif_start + motif_length]
        motif_ps = [pitch.Pitch(p) for p in motif_ps]
        motif_ps = [p.nameWithOctave for p in motif_ps]
        motif_ps = ' '.join(motif_ps)
        motifs.append(motif_ps)
    return motifs


def query_all():
    bound_lower = 5
    bound_upper = 16
    motifs = []
    g = graphviz.Digraph(name='motifs', directory='data', format='svg')
    for motif_length in range(bound_lower, bound_upper):
        print('Examining motifs of length', motif_length)
        motifs.append(query_any(motif_length))
        if len(motifs) > 1:
            for motif_sub in motifs[-2]:
                for motif_sup in motifs[-1]:
                    if motif_sub in motif_sup:
                        g.edge(motif_sub, motif_sup)
    g.view()


if __name__ == "__main__":
    query_all()

import os.path

import igraph as ig
import numpy as np
import regex as re
import stumpy
from music21 import pitch

from data import generate_input


def __lps__(pat: [float]) -> [int]:
    ret = [0]
    for i in range(1, len(pat)):
        j = ret[i - 1]
        while j > 0 and pat[j] != pat[i]:
            j = ret[j - 1]
        ret.append(j + 1 if pat[j] == pat[i] else j)
    return ret


def kmp(pat: [float], sig: [float]) -> [int]:
    lps, ret, j = __lps__(pat), [], 0
    for i in range(len(sig)):
        while j > 0 and sig[i] != pat[j]:
            j = lps[j - 1]
        if sig[i] == pat[j]:
            j += 1
        if j == len(pat):
            ret.append(i - (j - 1))
            j = lps[j - 1]
    return ret


def query_any(motif_length: int = 10) -> set:
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
    print(sig)

    if not os.path.exists('data/stump_{0}.npy'.format(motif_length)):
        sig_profile = stumpy.stump(T_A=sig,
                                   m=motif_length)
        np.save('data/stump_{0}'.format(motif_length), sig_profile, allow_pickle=True)
    sig_profile = np.load('data/stump_{0}.npy'.format(motif_length), allow_pickle=True)
    sig_motifs = stumpy.motifs(T=sig,
                               P=sig_profile[:, 0],
                               min_neighbors=5,
                               max_distance=0.0,
                               cutoff=None,
                               max_matches=10,
                               max_motifs=1000)

    motifs = set()
    for m in sig_motifs[1]:
        motif_start = m[0]
        motif_ps = sig[motif_start:motif_start + motif_length]
        print(kmp(motif_ps, sig))
        motif_ps = [pitch.Pitch(p) for p in motif_ps]
        motif_ps = [p.nameWithOctave for p in motif_ps]
        motif_ps = ' '.join(motif_ps)
        motifs.add(motif_ps)
    return motifs


def query_all():
    bound_lower = 5
    bound_upper = 16

    if not os.path.exists('data/motifs.gml'):
        generate_input()
        pitches = []
        with open('data/pitch_training.txt', 'rt') as file:
            pitches += file.read().split(' ')
        with open('data/pitch_validation.txt', 'rt') as file:
            pitches += file.read().split(' ')
        data = []
        for p in pitches:
            if p != 'RST':
                data.append(pitch.Pitch(p).ps)
        data = [pitch.Pitch(p) for p in data]
        data = [p.nameWithOctave for p in data]
        data = ' '.join(data)
        with open('data.txt', 'wt') as file:
            file.write(data)

        motifs = []
        for motif_length in range(bound_lower, bound_upper):
            print('Examining motifs of length', motif_length)
            motifs.append(list(query_any(motif_length)))

        vertices = 0
        offsets = []
        labels = []
        for motif_length in range(bound_lower, bound_upper):
            offsets.append(vertices)
            vertices += len(motifs[motif_length - bound_lower])
            labels += motifs[motif_length - bound_lower]

        for idx in range(len(labels)):
            occ = len(re.findall(pattern=re.escape(pattern=labels[idx],
                                                   special_only=True,
                                                   literal_spaces=False),
                                 string=data,
                                 overlapped=True))
            if occ == 1:
                print(labels[idx])

        edges = []
        for motif_length in range(bound_lower + 1, bound_upper):
            for idx_sub in range(len(motifs[motif_length - bound_lower - 1])):
                motif_sub = motifs[motif_length - bound_lower - 1][idx_sub]
                for idx_sup in range(len(motifs[motif_length - bound_lower])):
                    motif_sup = motifs[motif_length - bound_lower][idx_sup]
                    if motif_sub in motif_sup:
                        edges.append((offsets[motif_length - bound_lower - 1] + idx_sub, offsets[motif_length - bound_lower] + idx_sup))

        g = ig.Graph(n=vertices, edges=edges, directed=True)
        g.vs['label'] = labels
        g.save('data/motifs.gml')
    g = ig.load('data/motifs.gml')

    g_comp = g.connected_components(mode='weak')
    g_comp = [c for c in g_comp if len(c) > 1]
    g_comp.sort(key=len, reverse=True)
    for idx in range(10):
        ig.plot(g.subgraph(g_comp[idx]), target='motifs_{0}.png'.format(idx), bbox=(2048, 2048), margin=128)


if __name__ == '__main__':
    query_all()

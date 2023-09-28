import json
import os.path

import igraph as ig
import regex as re
from music21 import pitch
from tqdm import tqdm

from data import generate_input


def __lps__(pat: [float]) -> [int]:
    ret = [0]
    for i in range(1, len(pat)):
        j = ret[i - 1]
        while j > 0 and pat[j] != pat[i]:
            j = ret[j - 1]
        ret.append(j + 1 if pat[j] == pat[i] else j)
    return ret


def __hsh__(sig: [float], sz: int) -> [float]:
    ret = 0
    for i in range(sz):
        ret += sig[i]
    ret = [ret]
    for i in range(1, len(sig) - sz):
        ret.append(ret[-1] - sig[i - 1] + sig[i + sz - 1])
    return ret


def __kmp__(pat: [float], sig: [float]) -> [int]:
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
    if not os.path.exists('data/motifs_{0}.json'.format(motif_length)):
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

        motifs = set()
        for motif_start in tqdm(range(len(sig) - motif_length)):
            motif_ps = sig[motif_start:motif_start + motif_length]
            motif_occ = len(__kmp__(motif_ps, sig))
            if motif_occ > 4:
                motif_ps = [pitch.Pitch(p) for p in motif_ps]
                motif_ps = [p.nameWithOctave for p in motif_ps]
                motif_ps = ' '.join(motif_ps)
                motifs.add(motif_ps)

        motifs = list(motifs)
        with open('data/motifs_{0}.json'.format(motif_length), 'wt') as file:
            json.dump(motifs, file)
    with open('data/motifs_{0}.json'.format(motif_length), 'rt') as file:
        motifs = json.load(file)
    return set(motifs)


def query_all():
    bound_lower = 4
    bound_upper = 25

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
            assert occ > 4

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

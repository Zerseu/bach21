import json
import os.path

import igraph as ig
import regex as re
from music21 import pitch
from tqdm import tqdm

from data import generate_input


def query_any(motif_length: int = 10) -> set:
    if not os.path.exists('data/motifs_{0}.json'.format(motif_length)):
        generate_input()

        pitches = []
        with open('data/pitch_training.txt', 'rt') as file:
            pitches += file.read().split(' ')
        with open('data/pitch_validation.txt', 'rt') as file:
            pitches += file.read().split(' ')

        sig_raw = []
        for p in pitches:
            if p != 'RST':
                sig_raw.append(pitch.Pitch(p).ps)
        sig_str = ' '.join([pitch.Pitch(p).nameWithOctave for p in sig_raw])

        motifs = set()
        for motif_start in tqdm(range(len(sig_raw) - motif_length)):
            motif_raw = sig_raw[motif_start:motif_start + motif_length]
            motif_str = ' '.join([pitch.Pitch(p).nameWithOctave for p in motif_raw])
            motif_occ = len(re.findall(pattern=re.escape(pattern=motif_str,
                                                         special_only=True,
                                                         literal_spaces=False),
                                       string=sig_str,
                                       overlapped=True))
            if motif_occ > 4:
                motifs.add(motif_str + ' ' + str(motif_occ))

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

        edges = []
        for motif_length in range(bound_lower + 1, bound_upper):
            for idx_sub in range(len(motifs[motif_length - bound_lower - 1])):
                motif_sub = ' '.join(motifs[motif_length - bound_lower - 1][idx_sub].split(' ')[:-1])
                # motif_sub_occ = int(motifs[motif_length - bound_lower - 1][idx_sub].split(' ')[-1])
                for idx_sup in range(len(motifs[motif_length - bound_lower])):
                    motif_sup = ' '.join(motifs[motif_length - bound_lower][idx_sup].split(' ')[:-1])
                    # motif_sup_occ = int(motifs[motif_length - bound_lower][idx_sup].split(' ')[-1])
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
        ig.plot(obj=g.subgraph(g_comp[idx]),
                target='motifs_{0}.png'.format(idx),
                bbox=(2000, 2000),
                margin=100)


if __name__ == '__main__':
    query_all()

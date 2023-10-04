import json
import multiprocessing
import os.path
import sys

import igraph as ig
import regex as re
from music21 import pitch

from data import get_dir, generate_input


def query_any(composer: str, instruments: [str], motif_length: int) -> (int, {}):
    print('Examining motifs of length', motif_length)
    crt_dir = get_dir(composer, instruments)

    if not os.path.exists(os.path.join(crt_dir, 'motifs_{:02d}.json'.format(motif_length))):
        generate_input(composer, instruments)

        pitches = []
        with open(os.path.join(crt_dir, 'pitch_training.txt'), 'rt') as file:
            pitches += file.read().split(' ')
        with open(os.path.join(crt_dir, 'pitch_validation.txt'), 'rt') as file:
            pitches += file.read().split(' ')

        sig_raw = []
        for p in pitches:
            if p != 'RST':
                sig_raw.append(pitch.Pitch(p).ps)
        sig_str = ' '.join([pitch.Pitch(p).nameWithOctave for p in sig_raw])

        motifs = {}
        for motif_start in range(len(sig_raw) - motif_length):
            motif_raw = sig_raw[motif_start:motif_start + motif_length]
            motif_str = ' '.join([pitch.Pitch(p).nameWithOctave for p in motif_raw])
            if motif_str not in motifs:
                motif_occ = len(re.findall(pattern=re.escape(pattern=motif_str,
                                                             special_only=True,
                                                             literal_spaces=False),
                                           string=sig_str,
                                           overlapped=True))
                if motif_occ >= 4:
                    motifs[motif_str] = motif_occ

        with open(os.path.join(crt_dir, 'motifs_{:02d}.json'.format(motif_length)), 'wt') as file:
            json.dump(motifs, file, indent=4, sort_keys=True)
    with open(os.path.join(crt_dir, 'motifs_{:02d}.json'.format(motif_length)), 'rt') as file:
        return motif_length, json.load(file)


def query_all(composer: str, instruments: [str]):
    crt_dir = get_dir(composer, instruments)

    bound_lower = 4
    bound_upper = 25

    if not os.path.exists(os.path.join(crt_dir, 'motifs.gml')):
        generate_input(composer, instruments)
        pitches = []
        with open(os.path.join(crt_dir, 'pitch_training.txt'), 'rt') as file:
            pitches += file.read().split(' ')
        with open(os.path.join(crt_dir, 'pitch_validation.txt'), 'rt') as file:
            pitches += file.read().split(' ')

        print('Running parallel motif discovery from length', bound_lower, 'to', bound_upper - 1)
        motifs = {}
        with multiprocessing.Pool(4) as pool:
            args = [(composer, instruments, motif_length) for motif_length in range(bound_lower, bound_upper)]
            for result in pool.starmap(query_any, args):
                motifs[result[0]] = result[1]
        print('Motif discovery complete...')

        print('Computing motif relationship graph...')
        vertices = 0
        labels = []
        occurrences = []
        lengths = []
        for motif_length in range(bound_lower, bound_upper):
            vertices += len(motifs[motif_length])
            for motif in motifs[motif_length]:
                labels.append(motif)
                occurrences.append(motifs[motif_length][motif])
                lengths.append(motif_length)

        edges = []
        for motif_sub_idx in range(0, vertices - 1):
            for motif_sup_idx in range(motif_sub_idx + 1, vertices):
                if lengths[motif_sub_idx] + 1 == lengths[motif_sup_idx]:
                    if labels[motif_sub_idx] in labels[motif_sup_idx]:
                        edges.append((motif_sub_idx, motif_sup_idx))
        print('Graph computation complete...')

        g = ig.Graph(n=vertices, edges=edges, directed=True)
        g.vs['label'] = labels
        g.save(os.path.join(crt_dir, 'motifs.gml'))
    g = ig.load(os.path.join(crt_dir, 'motifs.gml'))

    print('Plotting motif relationship top-10 connected components...')
    g_comp = g.connected_components(mode='weak')
    g_comp = [c for c in g_comp if len(c) > 1]
    g_comp.sort(key=len, reverse=True)
    for idx in range(min(10, len(g_comp))):
        ig.plot(obj=g.subgraph(g_comp[idx]),
                target=os.path.join(crt_dir, 'graph_{:02d}.png'.format(idx)),
                bbox=(2000, 2000),
                margin=100)
    print('Component plot complete...')


if __name__ == '__main__':
    query_all(sys.argv[1], sys.argv[2:])

import json
import multiprocessing
import os.path
import sys

import igraph as ig
import networkx as nx
import regex as re
from music21 import pitch
from netrd.distance import LaplacianSpectral
from tqdm import tqdm

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
            if p != 'SIG' and p != 'RST':
                sig_raw.append(pitch.Pitch(p).ps)
        sig_str = ' '.join([pitch.Pitch(p).nameWithOctave for p in sig_raw])

        motifs = {}
        for motif_start in range(len(sig_raw) - motif_length):
            motif_raw = sig_raw[motif_start:motif_start + motif_length]
            motif_str = ' '.join([pitch.Pitch(p).nameWithOctave for p in motif_raw])
            if motif_str not in motifs:
                motif_occ = len(re.findall(pattern=re.escape(pattern=motif_str,
                                                             special_only=False,
                                                             literal_spaces=False),
                                           string=sig_str,
                                           overlapped=True))
                if motif_occ >= 5:
                    motifs[motif_str] = motif_occ

        with open(os.path.join(crt_dir, 'motifs_{:02d}.json'.format(motif_length)), 'wt') as file:
            json.dump(motifs, file, indent=4, sort_keys=True)
    with open(os.path.join(crt_dir, 'motifs_{:02d}.json'.format(motif_length)), 'rt') as file:
        return motif_length, json.load(file)


def query_all(composer: str, instruments: [str]):
    crt_dir = get_dir(composer, instruments)

    bound_lower_inc = 8
    bound_upper_inc = 24

    if not os.path.exists(os.path.join(crt_dir, 'motifs.gml')):
        generate_input(composer, instruments)
        pitches = []
        with open(os.path.join(crt_dir, 'pitch_training.txt'), 'rt') as file:
            pitches += file.read().split(' ')
        with open(os.path.join(crt_dir, 'pitch_validation.txt'), 'rt') as file:
            pitches += file.read().split(' ')

        print('Running parallel motif discovery from length', bound_lower_inc, 'to', bound_upper_inc)
        motifs = {}
        with multiprocessing.Pool(4) as pool:
            args = [(composer, instruments, motif_length) for motif_length in range(bound_lower_inc, bound_upper_inc + 1)]
            for result in pool.starmap(query_any, args):
                motifs[result[0]] = result[1]
        print('Motif discovery complete...')

        print('Computing motif relationship graph...')
        vertices = 0
        labels = []
        occurrences = []
        lengths = []
        for motif_length in range(bound_lower_inc, bound_upper_inc + 1):
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

        g = ig.Graph(n=vertices,
                     edges=edges,
                     directed=True)
        g.vs['label'] = labels
        g.vs['occurrence'] = occurrences
        g.vs['length'] = lengths
        g.save(os.path.join(crt_dir, 'motifs.gml'))
    g = ig.load(os.path.join(crt_dir, 'motifs.gml'))

    print('Plotting motif relationship top-10 connected components...')
    g_comp = g.connected_components(mode='weak')
    g_comp = [c for c in g_comp if len(c) > 1]
    g_comp.sort(key=len, reverse=True)
    for idx in tqdm(range(min(10, len(g_comp)))):
        conn_comp = g.subgraph(g_comp[idx])
        conn_comp.save(os.path.join(crt_dir, 'layout_{:02d}.gml'.format(idx)))

        ig.plot(obj=conn_comp,
                target=os.path.join(crt_dir, 'layout_{:02d}.svg'.format(idx)),
                palette=ig.GradientPalette('green', 'red'),
                vertex_size=25,
                vertex_color=list(map(int, ig.rescale(values=g.subgraph(g_comp[idx]).vs['length'],
                                                      out_range=(0, 255),
                                                      clamp=True))),
                bbox=(2 ** 13, 2 ** 13),
                margin=2 ** 7)

        conn_comp_lout = conn_comp.layout('auto')
        with open(os.path.join(crt_dir, 'layout_{:02d}.json'.format(idx)), 'wt') as file:
            json.dump(conn_comp_lout.coords, file, indent=4, sort_keys=False)
    print('Component plot complete...')


def query_distance(composer1: str, instruments1: [str],
                   composer2: str, instruments2: [str]) -> float:
    query_all(composer1, instruments1)
    query_all(composer2, instruments2)

    dir1 = get_dir(composer1, instruments1)
    dir2 = get_dir(composer2, instruments2)

    g1 = nx.read_gml(os.path.join(dir1, 'motifs.gml'))
    g2 = nx.read_gml(os.path.join(dir2, 'motifs.gml'))

    dist_obj = LaplacianSpectral()
    return dist_obj.dist(g1, g2)


if __name__ == '__main__':
    query_all(sys.argv[1], sys.argv[2:])
    # print(query_distance('bach', ['violin'], 'bach', ['flute']))

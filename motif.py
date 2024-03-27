import json
import multiprocessing
import os.path

import igraph as ig
import networkx as nx
import regex as re
from netrd.distance.laplacian_spectral_method import LaplacianSpectral
from tqdm import tqdm

from data import get_dir, generate_input

LengthLowerBound: int = 8
LengthUpperBound: int = 24


def query_any(composer: str, instruments: [str], motif_length: int) -> (int, {}):
    print('Examining motifs of length', motif_length)
    crt_dir = get_dir(composer, instruments)

    if not os.path.exists(os.path.join(crt_dir, 'motifs_{:02d}.json'.format(motif_length))):
        generate_input(composer, instruments)

        sentences = []
        total_units = 0
        with open(os.path.join(crt_dir, 'pitch_input.txt'), 'rt') as file:
            for sentence in file.read().split('\n'):
                sentences.append([])
                for word in sentence.split():
                    if word != 'RST':
                        sentences[-1].append(word)
                        total_units += 1

        motifs: dict[str, int] = {}
        visited: set[str] = set()

        assert total_units >= 100
        assert LengthLowerBound <= motif_length <= LengthUpperBound

        for sentence in sentences:
            sentence_str = ' '.join(sentence)
            for motif_start in range(len(sentence) - motif_length):
                motif_str = ' '.join(sentence[motif_start:motif_start + motif_length])
                if motif_str in visited:
                    continue
                visited.add(motif_str)

                if motif_str not in motifs:
                    motif_occ = len(re.findall(pattern=re.escape(pattern=motif_str,
                                                                 special_only=False,
                                                                 literal_spaces=False),
                                               string=sentence_str,
                                               overlapped=True))
                    if motif_occ >= 2:
                        if (motif_occ * motif_length) / total_units >= 0.0005:
                            motifs[motif_str] = motif_occ

        with open(os.path.join(crt_dir, 'motifs_{:02d}.json'.format(motif_length)), 'wt') as file:
            json.dump(motifs, file, indent=4, sort_keys=True)
    with open(os.path.join(crt_dir, 'motifs_{:02d}.json'.format(motif_length)), 'rt') as file:
        return motif_length, json.load(file)


def query_all(composer: str, instruments: [str], plot: bool = True):
    crt_dir = get_dir(composer, instruments)

    if not os.path.exists(os.path.join(crt_dir, 'motifs.gml')):
        generate_input(composer, instruments)

        pitches = []
        with open(os.path.join(crt_dir, 'pitch_input.txt'), 'rt') as file:
            for word in file.read().split():
                pitches.append(word)

        print('Running parallel motif discovery from length', LengthLowerBound, 'to', LengthUpperBound)
        motifs = {}
        with multiprocessing.Pool(multiprocessing.cpu_count() - 4) as pool:
            args = [(composer, instruments, motif_length) for motif_length in range(LengthLowerBound, LengthUpperBound + 1)]
            for result in pool.starmap(query_any, args):
                motifs[result[0]] = result[1]
        print('Motif discovery complete...')

        print('Computing motif relationship graph...')
        vertices = 0
        labels = []
        occurrences = []
        lengths = []
        for motif_length in range(LengthLowerBound, LengthUpperBound + 1):
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

    if plot:
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
    query_all(composer1, instruments1, False)
    query_all(composer2, instruments2, False)

    dir1 = get_dir(composer1, instruments1)
    dir2 = get_dir(composer2, instruments2)

    g1 = nx.read_gml(os.path.join(dir1, 'motifs.gml'))
    g2 = nx.read_gml(os.path.join(dir2, 'motifs.gml'))

    dist_obj = LaplacianSpectral()
    return dist_obj.dist(g1, g2)


def main():
    composers = ['bach', 'beethoven', 'mozart', 'paganini', 'vivaldi']
    instruments = []
    for composer in composers:
        generate_input(composer, instruments)

    with open('distances.csv', 'wt') as csv:
        csv.write('Composer1,Composer2,Distance\n')
        for composer1 in composers:
            for composer2 in composers:
                distance = query_distance(composer1, instruments, composer2, instruments)
                csv.write('{},{},{}\n'.format(composer1, composer2, distance))


if __name__ == '__main__':
    main()

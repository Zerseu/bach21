import json
import os.path
import random
from typing import Optional

from music21 import *
from tqdm import tqdm

InternalCorpus: bool = True
ExternalCorpus: Optional[str] = 'C:\\midi'
CorpusCache: str = 'bach21db'
FilterMatches: bool = True

if ExternalCorpus is not None:
    corpus.addPath(ExternalCorpus)


def get_dir(composer: str, instruments: [str]) -> str:
    if len(instruments) == 0:
        crt_dir = os.path.join(CorpusCache, composer, 'all')
    else:
        crt_dir = os.path.join(CorpusCache, composer, instruments[0])
    if not os.path.exists(crt_dir):
        os.makedirs(crt_dir)
    return crt_dir


def lcs(a: [float], b: [float]) -> int:
    m = len(a)
    n = len(b)
    aux = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    ret = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                aux[i][j] = 0
            elif a[i - 1] == b[j - 1]:
                aux[i][j] = aux[i - 1][j - 1] + 1
                ret = max(ret, aux[i][j])
            else:
                aux[i][j] = 0
    return ret


def valid_part(part, instruments: [str]) -> bool:
    if part is None:
        return False
    if len(instruments) == 0:
        return True
    else:
        instr = part.getInstrument()
        if instr is None:
            return False
        name = instr.instrumentName
        if name is None:
            return False
        ok = False
        for instr in instruments:
            if instr.lower() in name.lower():
                ok = True
                break
        return ok


def unravel_part(part) -> (str, [float], [float]):
    if part is None:
        return None
    part_instrument = part.getInstrument()
    if part_instrument is None:
        return None
    part_instrument = part_instrument.bestName()
    if part_instrument is None:
        return None
    if part_instrument == "":
        return None
    part_instrument = part_instrument.lower()
    part_pitches = []
    part_durations = []
    for element in part.flatten().getElementsByClass([note.Note, note.Rest]):
        part_durations.append(element.duration.quarterLengthNoTuplets)
        if element.isNote:
            part_pitches.append(element.pitch.ps)
        if element.isRest:
            part_pitches.append(0)
    if len(part_pitches) == 0 or len(part_durations) == 0:
        return None
    assert len(part_pitches) == len(part_durations)
    return part_instrument, part_pitches, part_durations


def rebuild_corpus_cache():
    cache_dir = os.path.join(CorpusCache, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    corpus_type = []
    if InternalCorpus is True:
        corpus_type.append('core')
    if ExternalCorpus is not None:
        corpus_type.append('local')
    for pth in tqdm(corpus.getPaths(name=corpus_type)):
        composition = corpus.parse(pth)
        cache_file = os.path.splitext(composition.metadata.corpusFilePath)[0]
        if ExternalCorpus is not None and cache_file.startswith(ExternalCorpus):
            cache_file = cache_file[len(ExternalCorpus) + 1:]
        cache_file_clean = ''
        for ch in cache_file:
            if ch.isalnum():
                cache_file_clean += ch
            else:
                cache_file_clean += '_'
        while cache_file != cache_file_clean:
            cache_file = cache_file_clean
            cache_file_clean = cache_file.replace('__', '_')
        cache_file = cache_file.lower()
        cache_file = os.path.join(cache_dir, cache_file + '.json')
        cache_dict = {}
        parts = instrument.partitionByInstrument(composition)
        for part in parts:
            p = unravel_part(part)
            if p is not None:
                if p[0] not in cache_dict:
                    cache_dict[p[0]] = [[], []]
                cache_dict[p[0]][0] += p[1]
                cache_dict[p[0]][1] += p[2]
        if len(cache_dict) > 0:
            with open(cache_file, 'wt') as file:
                json.dump(cache_dict, file)


def generate_input(composer: str, instruments: [str], ratio: float = 0.8):
    crt_dir = get_dir(composer, instruments)

    pth_pitch_training = os.path.join(crt_dir, 'pitch_training.txt')
    pth_pitch_validation = os.path.join(crt_dir, 'pitch_validation.txt')

    pth_duration_training = os.path.join(crt_dir, 'duration_training.txt')
    pth_duration_validation = os.path.join(crt_dir, 'duration_validation.txt')

    if os.path.exists(pth_pitch_training) and os.path.exists(pth_pitch_validation):
        if os.path.exists(pth_duration_training) and os.path.exists(pth_duration_validation):
            return

    print('Generating input data...')
    matches = []

    if InternalCorpus is True:
        print('Parsing internal corpus -', 'music21')
        for composition in tqdm(corpus.search(composer, 'composer')):
            parts = instrument.partitionByInstrument(corpus.parse(composition))
            for part in parts:
                if valid_part(part, instruments):
                    matches.append(part)
        print('Done parsing internal corpus...')

    if ExternalCorpus is not None:
        print('Parsing external corpus -', ExternalCorpus)
        for root, dirs, files in tqdm(os.walk(ExternalCorpus)):
            for pth in sorted(files):
                pth = os.path.join(root, pth)
                if pth.endswith('.mid'):
                    if composer.lower() in pth.lower():
                        parts = instrument.partitionByInstrument(converter.parse(value=pth, format='midi'))
                        for part in parts:
                            if valid_part(part, instruments):
                                matches.append(part)
        print('Done parsing external corpus...')

    if FilterMatches:
        print('Excluding duplicate parts...')
        pitches = []
        durations = []
        for part in matches:
            pitches.append([])
            durations.append([])
            for element in part.flatten().getElementsByClass([note.Note, note.Rest]):
                durations[-1].append(element.duration.quarterLength)
                if element.isNote:
                    pitches[-1].append(element.pitch.ps)
                if element.isRest:
                    pitches[-1].append(0)

        done = False
        while not done:
            done = True
            for idx in range(len(matches) - 1):
                if len(pitches[idx]) < len(pitches[idx + 1]):
                    aux = pitches[idx]
                    pitches[idx] = pitches[idx + 1]
                    pitches[idx + 1] = aux
                    aux = durations[idx]
                    durations[idx] = durations[idx + 1]
                    durations[idx + 1] = aux
                    aux = matches[idx]
                    matches[idx] = matches[idx + 1]
                    matches[idx + 1] = aux
                    done = False

        invalid = 0
        valid = [True]
        for idx in tqdm(range(1, len(matches))):
            ok = len(pitches[idx]) > 0 and len(durations[idx]) > 0
            ok = ok and sum(p == 0 for p in pitches[idx]) / len(pitches[idx]) < 0.25

            if ok:
                for idy in range(idx):
                    if valid[idy]:
                        if lcs(pitches[idx], pitches[idy]) / min(len(pitches[idx]), len(pitches[idy])) > 0.75:
                            ok = False
                            break
            if not ok:
                invalid += 1
            valid.append(ok)
        matches = [matches[idx] for idx in range(len(matches)) if valid[idx]]
        print('Excluded', invalid, 'parts!')
        print('Done excluding duplicate parts...')

    random.shuffle(matches)

    pitches = []
    durations = []
    for part in matches:
        for element in part.flatten().getElementsByClass([note.Note, note.Rest]):
            durations.append(str(element.duration.quarterLength))
            if element.isNote:
                pitches.append(str(element.nameWithOctave))
            if element.isRest:
                pitches.append('RST')
        durations.append('SIG')
        pitches.append('SIG')
    assert len(pitches) == len(durations)

    length = len(pitches)
    split_point = int(length * ratio)

    with open(pth_pitch_training, 'wt') as file:
        file.write(' '.join(pitches[:split_point]))
    with open(pth_pitch_validation, 'wt') as file:
        file.write(' '.join(pitches[split_point:]))

    with open(pth_duration_training, 'wt') as file:
        file.write(' '.join(durations[:split_point]))
    with open(pth_duration_validation, 'wt') as file:
        file.write(' '.join(durations[split_point:]))
    print('Done generating input data...')


def generate_output(composer: str, instruments: [str]):
    crt_dir = get_dir(composer, instruments)

    pth_pitch = os.path.join(crt_dir, 'pitch_output.txt')
    pth_duration = os.path.join(crt_dir, 'duration_output.txt')
    pth_midi = os.path.join(crt_dir, 'midi_output.mid')
    pth_xml = os.path.join(crt_dir, 'xml_output.musicxml')

    print('Generating output data...')
    with open(pth_pitch, 'rt') as file:
        pitches = file.read().split(' ')
    with open(pth_duration, 'rt') as file:
        durations = file.read().split(' ')
    assert len(pitches) == len(durations)
    length = len(pitches)

    composition = stream.Stream()
    composition.append(clef.TrebleClef())
    composition.append(instrument.Violin())
    for i in range(length):
        try:
            length = float(durations[i])
        except ValueError:
            length = 0
        if length <= 0:
            continue

        if pitches[i] == 'SIG':
            continue
        elif pitches[i] == 'RST':
            element = note.Rest(length)
        else:
            element = note.Note(pitches[i])
            element.duration.quarterLength = length
        composition.append(element)
    composition.makeMeasures(inPlace=True)
    composition.write('midi', pth_midi)
    composition.write('musicxml', pth_xml)
    print('Done generating output data...')

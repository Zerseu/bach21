import json
import os.path

import music21.note
from music21 import *
from tqdm import tqdm

DataRoot: str = 'bach21data'
FilterMatches: bool = True


def get_dir(composer: str, instruments: [str]) -> str:
    if len(instruments) == 0:
        crt_dir = os.path.join(DataRoot, composer, 'all')
    else:
        crt_dir = os.path.join(DataRoot, composer, instruments[0])
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
    num_pitches = []
    num_durations = []

    print('Parsing cached corpus...')
    for root, dirs, files in tqdm(os.walk('bach21cache')):
        for file_pth in sorted(files):
            full_pth = os.path.join(root, file_pth)
            if full_pth.endswith('.json'):
                if composer.lower() in file_pth.lower():
                    with open(full_pth, 'rt') as file:
                        parts = json.load(file)
                        for part in parts:
                            for instr in instruments:
                                if instr.lower() in part.lower():
                                    num_pitches.append(parts[part][0])
                                    num_durations.append(parts[part][1])
    print('Done parsing cached corpus...')

    if FilterMatches:
        print('Excluding duplicate parts...')
        length = len(num_pitches)
        done = False
        while not done:
            done = True
            for idx in range(length - 1):
                if len(num_pitches[idx]) < len(num_pitches[idx + 1]):
                    aux = num_pitches[idx]
                    num_pitches[idx] = num_pitches[idx + 1]
                    num_pitches[idx + 1] = aux
                    aux = num_durations[idx]
                    num_durations[idx] = num_durations[idx + 1]
                    num_durations[idx + 1] = aux
                    done = False

        invalid = 0
        valid = [True]
        for idx in tqdm(range(1, length)):
            ok = len(num_pitches[idx]) > 0 and len(num_durations[idx]) > 0
            ok = ok and sum(p == 0 for p in num_pitches[idx]) / len(num_pitches[idx]) < 0.25

            if ok:
                for idy in range(idx):
                    if valid[idy]:
                        if lcs(num_pitches[idx], num_pitches[idy]) / min(len(num_pitches[idx]), len(num_pitches[idy])) > 0.75:
                            ok = False
                            break
            if not ok:
                invalid += 1
            valid.append(ok)
        num_pitches = [num_pitches[idx] for idx in range(length) if valid[idx]]
        num_durations = [num_durations[idx] for idx in range(length) if valid[idx]]
        print('Excluded', invalid, 'parts!')
        print('Done excluding duplicate parts...')

    # random.shuffle(matches)

    str_pitches = []
    for pitches in num_pitches:
        for element in pitches:
            if element == 0:
                str_pitches.append('RST')
            else:
                str_pitches.append(str(music21.note.Note(element).nameWithOctave))
        str_pitches.append('SIG')

    str_durations = []
    for durations in num_durations:
        for element in durations:
            str_durations.append(str(music21.duration.Duration(element).quarterLength))
        str_durations.append('SIG')

    assert len(str_pitches) == len(str_durations)

    length = min(len(str_pitches), len(str_durations))
    split_point = int(length * ratio)

    with open(pth_pitch_training, 'wt') as file:
        file.write(' '.join(str_pitches[:split_point]))
    with open(pth_pitch_validation, 'wt') as file:
        file.write(' '.join(str_pitches[split_point:]))

    with open(pth_duration_training, 'wt') as file:
        file.write(' '.join(str_durations[:split_point]))
    with open(pth_duration_validation, 'wt') as file:
        file.write(' '.join(str_durations[split_point:]))
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

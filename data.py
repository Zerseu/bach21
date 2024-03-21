import json
import os.path
import random

import music21.note
from music21 import *
from tqdm import tqdm

DataRoot: str = 'bach21data'
FilterParts: bool = False
FilterRests: bool = True
random.seed(0)


def get_dir(composer: str, instruments: [str]) -> str:
    if len(instruments) == 0:
        crt_dir = os.path.join(DataRoot, composer, 'all')
    else:
        crt_dir = os.path.join(DataRoot, composer, instruments[0])
    if not os.path.exists(crt_dir):
        os.makedirs(crt_dir)
    return crt_dir


def lcs(a: [float], b: [float]) -> [float]:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return []
    dp = [[0] * (n + 1) for _ in range(2)]
    length = 0
    end_index = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
                if dp[i % 2][j] > length:
                    length = dp[i % 2][j]
                    end_index = i
            else:
                dp[i % 2][j] = 0
    if length == 0:
        return []
    return a[end_index - length: end_index]


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
    for root, dirs, files in os.walk('bach21cache'):
        random.shuffle(files)
        for file_pth in files:
            full_pth = os.path.join(root, file_pth)
            if full_pth.endswith('.json'):
                if composer.lower() in file_pth.lower():
                    with open(full_pth, 'rt') as file:
                        parts = json.load(file)
                        for part in parts:
                            for instr in instruments:
                                if instr.lower() in part.lower() or instr.lower() in file_pth.lower():
                                    num_pitches.append(parts[part][0])
                                    num_durations.append(parts[part][1])
    print('Done parsing cached corpus...')

    if FilterParts:
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
                        if len(lcs(num_pitches[idx], num_pitches[idy])) / min(len(num_pitches[idx]), len(num_pitches[idy])) > 0.75:
                            ok = False
                            break
            if not ok:
                invalid += 1
            valid.append(ok)
        num_pitches = [num_pitches[idx] for idx in range(length) if valid[idx]]
        num_durations = [num_durations[idx] for idx in range(length) if valid[idx]]
        print('Excluded', invalid, 'parts!')
        print('Done excluding duplicate parts...')

    train_pitches = []
    train_durations = []
    test_pitches = []
    test_durations = []

    for pitches, durations in zip(num_pitches, num_durations):
        str_pitches = []
        str_durations = []
        for e_pitch, e_duration in zip(pitches, durations):
            if e_pitch == 0:
                if FilterRests:
                    continue
                else:
                    str_pitches.append('RST')
            else:
                str_pitches.append(str(music21.note.Note(e_pitch).nameWithOctave))
            str_durations.append(str(music21.duration.Duration(e_duration).quarterLength))

        split_point = int(min(len(str_pitches), len(str_durations)) * ratio)
        train_pitches += str_pitches[:split_point]
        train_durations += str_durations[:split_point]
        test_pitches += str_pitches[split_point:]
        test_durations += str_durations[split_point:]

    assert len(train_pitches) == len(train_durations)
    assert len(test_pitches) == len(test_durations)

    with open(pth_pitch_training, 'wt') as file:
        file.write(' '.join(train_pitches))
    with open(pth_pitch_validation, 'wt') as file:
        file.write(' '.join(test_pitches))

    with open(pth_duration_training, 'wt') as file:
        file.write(' '.join(train_durations))
    with open(pth_duration_validation, 'wt') as file:
        file.write(' '.join(test_durations))
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

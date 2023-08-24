import os.path

from music21 import *


def generate_input(ratio: float = 0.8):
    pth_pitch_training = 'data/pitch_training.txt'
    pth_pitch_validation = 'data/pitch_validation.txt'

    pth_duration_training = 'data/duration_training.txt'
    pth_duration_validation = 'data/duration_validation.txt'

    if os.path.exists(pth_pitch_training) and os.path.exists(pth_pitch_validation):
        if os.path.exists(pth_duration_training) and os.path.exists(pth_duration_validation):
            return

    violin_parts = []
    for composition in corpus.search('bach', 'composer'):
        parts = instrument.partitionByInstrument(corpus.parse(composition))
        for part in parts:
            best_name = part.getInstrument().bestName()
            if 'Violin' in best_name or 'Soprano' in best_name:
                violin_parts.append(part)
    pitches = []
    durations = []
    for part in violin_parts:
        # pitches.append('BEG')
        # durations.append('BEG')
        for element in part.getElementsByClass(['Note', 'Rest']):
            durations.append(str(element.duration.quarterLength))
            if element.isNote:
                pitches.append(str(element.nameWithOctave))
            if element.isRest:
                pitches.append('RST')
        # pitches.append('END')
        # durations.append('END')
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
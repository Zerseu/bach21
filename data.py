import os.path

from music21 import *


def generate_input(composer: str = 'bach', ratio: float = 0.8, ):
    pth_pitch_training = 'data/pitch_training.txt'
    pth_pitch_validation = 'data/pitch_validation.txt'

    pth_duration_training = 'data/duration_training.txt'
    pth_duration_validation = 'data/duration_validation.txt'

    if os.path.exists(pth_pitch_training) and os.path.exists(pth_pitch_validation):
        if os.path.exists(pth_duration_training) and os.path.exists(pth_duration_validation):
            return

    print('Generating input data...')
    matches = []
    for composition in corpus.search(composer, 'composer'):
        parts = instrument.partitionByInstrument(corpus.parse(composition))
        for part in parts:
            best_name = part.getInstrument().bestName()
            if 'Violin' in best_name or 'Soprano' in best_name:
                matches.append(part)
    pitches = []
    durations = []
    for part in matches:
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
    print('Done generating input data...')


def generate_output():
    pth_pitch = 'data/pitch_output.txt'
    pth_duration = 'data/duration_output.txt'
    pth_midi = 'data/midi_output.mid'
    pth_xml = 'data/xml_output.musicxml'

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

        if pitches[i] == 'RST':
            element = note.Rest(length)
        else:
            element = note.Note(pitches[i])
            element.duration.quarterLength = length
        composition.append(element)
    composition.makeMeasures(inPlace=True)
    # composition.makeTies(inPlace=True)
    composition.write('midi', pth_midi)
    composition.write('musicxml', pth_xml)
    print('Done generating output data...')

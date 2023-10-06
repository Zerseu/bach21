import os.path
from typing import Optional

from music21 import *

InternalCorpus: bool = True
ExternalCorpus: Optional[str] = 'C:/midi'


def get_dir(composer: str, instruments: [str]) -> str:
    if len(instruments) == 0:
        crt_dir = os.path.join('data', composer, 'all')
    else:
        crt_dir = os.path.join('data', composer, instruments[0])
    if not os.path.exists(crt_dir):
        os.makedirs(crt_dir)
    return crt_dir


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
        print('Parsing internal corpus...')
        print('music21')
        for composition in corpus.search(composer, 'composer'):
            parts = instrument.partitionByInstrument(corpus.parse(composition))
            for part in parts:
                if len(instruments) == 0:
                    matches.append(part)
                else:
                    best_name = part.getInstrument().bestName()
                    ok = False
                    for inst in instruments:
                        if inst.lower() in best_name.lower():
                            ok = True
                            break
                    if ok:
                        matches.append(part)
        print('Done parsing internal corpus...')

    if ExternalCorpus is not None:
        print('Parsing external corpus...')
        print(ExternalCorpus)
        for root, dirs, files in os.walk(ExternalCorpus):
            if len(files) > 0:
                for pth in sorted(files):
                    pth = os.path.join(root, pth)
                    if pth.endswith('.mid'):
                        if composer.lower() in pth.lower():
                            ok = False
                            for inst in instruments:
                                if inst.lower() in pth.lower():
                                    ok = True
                                    break
                            if ok:
                                print('Parsing', pth)
                                parts = instrument.partitionByInstrument(converter.parse(value=pth, format='midi'))
                                for part in parts:
                                    matches.append(part)
        print('Done parsing external corpus...')

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

from music21 import *


def generate_input():
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
        pitches.append('BEG')
        durations.append('BEG')
        for element in part.getElementsByClass(['Note', 'Rest']):
            durations.append(str(element.duration.quarterLength))
            if element.isNote:
                pitches.append(str(element.nameWithOctave))
            if element.isRest:
                pitches.append('RST')
        pitches.append('END')
        durations.append('END')
    assert len(pitches) == len(durations)
    print(len(pitches))
    with open('pitches.txt', 'wt') as file:
        file.write(' '.join(pitches))
    with open('durations.txt', 'wt') as file:
        file.write(' '.join(durations))

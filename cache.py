import json
import os.path
from typing import Optional

from music21 import *
from tqdm import tqdm

from config import log

InternalCorpus: bool = True
ExternalCorpus: Optional[str] = 'C:\\midi'
CacheRoot: str = 'bach21cache'

if ExternalCorpus is not None:
    corpus.addPath(ExternalCorpus)


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


def rebuild_cache():
    composer_dict = {}
    if not os.path.exists(CacheRoot):
        os.makedirs(CacheRoot)
    corpus_type = []
    if InternalCorpus is True:
        corpus_type.append('core')
    if ExternalCorpus is not None:
        corpus_type.append('local')
    for pth in tqdm(corpus.getPaths(name=corpus_type)):
        try:
            composition = corpus.parse(pth)
        except Exception:
            log('Skipping', pth, 'because it is corrupt...')
            continue
        file_name = os.path.splitext(composition.metadata.corpusFilePath)[0]
        if ExternalCorpus is not None and file_name.startswith(ExternalCorpus):
            file_name = file_name[len(ExternalCorpus) + 1:]
        file_name = file_name.replace('\\', '/')
        composer = file_name.split('/')[0].lower()
        if composer not in composer_dict:
            composer_dict[composer] = 0
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
            composer_dict[composer] += 1
            cache_file = f'{composer}_{composer_dict[composer]:04d}'
            cache_file = os.path.join(CacheRoot, cache_file + '.json')
            with open(cache_file, 'wt') as file:
                json.dump(cache_dict, file)


if __name__ == '__main__':
    rebuild_cache()

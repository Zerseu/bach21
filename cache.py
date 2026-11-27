"""Builds the corpus cache: one JSON file per score with the pitches and durations of each instrument.

The cache in ``bach21cache/`` is what the rest of the pipeline reads. A ready-made copy of it ships as
``bach21cache.zip`` and is unpacked automatically on first use (see ``ensure_cache``), so rebuilding it
is only needed to add more music:

    python cache.py                      # music21's built-in corpus only
    python cache.py --external C:\\midi   # plus your own folder of MIDI / MusicXML files

Each cache file is named ``<composer>_<NNNN>.json`` and maps an instrument name to
``[pitches, durations]``: MIDI pitch numbers (0 for a rest) and durations in quarter notes.
"""
import argparse
import json
import os.path
import zipfile
from typing import Optional

from music21 import corpus, instrument, note
from tqdm import tqdm

from config import CACHE_ROOT, CACHE_ZIP, ROOT, log


def ensure_cache():
    """Unpacks bach21cache.zip into the project folder when the cache folder does not exist yet."""
    if os.path.isdir(CACHE_ROOT):
        return
    if not os.path.exists(CACHE_ZIP):
        raise FileNotFoundError(f'Neither {CACHE_ROOT} nor {CACHE_ZIP} exists. Run "python cache.py" to build the cache.')
    log(f'Unpacking {os.path.basename(CACHE_ZIP)}...')
    with zipfile.ZipFile(CACHE_ZIP) as archive:
        archive.extractall(ROOT)


def unravel_part(part) -> Optional[tuple[str, list[float], list[float]]]:
    """Returns (instrument name, pitches, durations) for one part, or None when it has no named instrument
    or no notes. Rests get pitch 0."""
    if part is None:
        return None
    part_instrument = part.getInstrument()
    if part_instrument is None:
        return None
    part_instrument = part_instrument.bestName()
    if not part_instrument:
        return None
    part_pitches = []
    part_durations = []
    for element in part.flatten().getElementsByClass([note.Note, note.Rest]):
        part_durations.append(element.duration.quarterLengthNoTuplets)
        part_pitches.append(element.pitch.ps if element.isNote else 0)
    if len(part_pitches) == 0:
        return None
    return part_instrument.lower(), part_pitches, part_durations


def rebuild_cache(internal_corpus: bool = True, external_corpus: Optional[str] = None):
    """Parses every score of the chosen corpora and writes the cache files.

    internal_corpus: include music21's built-in ("core") corpus.
    external_corpus: optional folder of extra scores; its first sub-folder level is the composer name.
    """
    os.makedirs(CACHE_ROOT, exist_ok=True)
    corpus_type = []
    if internal_corpus:
        corpus_type.append('core')
    if external_corpus is not None:
        corpus.addPath(external_corpus)
        corpus_type.append('local')

    composer_count: dict[str, int] = {}
    for pth in tqdm(corpus.getPaths(name=corpus_type)):
        try:
            composition = corpus.parse(pth)
        except Exception:
            log('Skipping', pth, 'because it is corrupt...')
            continue
        file_name = os.path.splitext(composition.metadata.corpusFilePath)[0]
        if external_corpus is not None and file_name.startswith(external_corpus):
            file_name = file_name[len(external_corpus) + 1:]
        composer = file_name.replace('\\', '/').split('/')[0].lower()

        cache_dict: dict[str, list[list[float]]] = {}
        for part in instrument.partitionByInstrument(composition):
            unravelled = unravel_part(part)
            if unravelled is not None:
                name, pitches, durations = unravelled
                entry = cache_dict.setdefault(name, [[], []])
                entry[0] += pitches
                entry[1] += durations
        if len(cache_dict) > 0:
            composer_count[composer] = composer_count.get(composer, 0) + 1
            cache_file = os.path.join(CACHE_ROOT, f'{composer}_{composer_count[composer]:04d}.json')
            with open(cache_file, 'wt') as file:
                json.dump(cache_dict, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rebuild the bach21cache corpus cache.')
    parser.add_argument('--external', metavar='DIR', default=None,
                        help='folder with extra MIDI / MusicXML scores, one sub-folder per composer')
    parser.add_argument('--no-internal', action='store_true', help="leave out music21's built-in corpus")
    arguments = parser.parse_args()
    rebuild_cache(internal_corpus=not arguments.no_internal, external_corpus=arguments.external)

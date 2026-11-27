"""Turns the corpus cache into token files for training, and generated tokens back into a score.

For a composer and instrument (for example ``bach`` and ``violin``) ``generate_input`` writes, in
``bach21data/<composer>/<instrument>/`` (``all`` instead of the instrument when none is given):

- ``note_input.txt``: combined pitch+duration tokens such as ``C5_0.5`` or ``RST_1``, one part per line.
  This is what the note model trains on.
- ``pitch_input.txt`` and ``duration_input.txt``: the same parts as separate pitch and duration
  streams (rests left out). entropy.py reads the pitch file.

``generate_output`` turns ``note_output.txt`` (or ``pitch_output.txt`` + ``duration_output.txt``) into
``midi_output.mid`` and ``xml_output.musicxml``.
"""
import json
import os.path
import random

from music21 import clef, duration, instrument, note, stream
from tqdm import tqdm

from cache import ensure_cache
from config import CACHE_ROOT, DATA_ROOT, log

FILTER_PARTS: bool = False      # drop near-duplicate parts and parts with many rests (slow, off by default)
FILTER_RESTS: bool = True       # leave rests out of pitch_input.txt / duration_input.txt
DURATION_GRID: float = 0.125    # note token durations are rounded to multiples of a 32nd note
MAX_REST_LENGTH: float = 4.0    # longer rests are shortened to this many quarter notes
TOKEN_SEPARATOR: str = '_'
REST_TOKEN: str = 'RST'

# Names kept for older code that imports them.
RestToken = REST_TOKEN
TokenSeparator = TOKEN_SEPARATOR


def quantize_duration(in_duration: float) -> float:
    """Rounds a duration to the nearest multiple of DURATION_GRID, never below one grid step."""
    steps = max(1, round(float(in_duration) / DURATION_GRID))
    return steps * DURATION_GRID


def make_note_token(pitch: str, quarter_length: float) -> str:
    return f'{pitch}{TOKEN_SEPARATOR}{quarter_length:g}'


def split_note_token(token: str) -> tuple[str, float]:
    pitch, quarter_length = token.rsplit(TOKEN_SEPARATOR, 1)
    return pitch, float(quarter_length)


def pitch_name(midi_pitch: float) -> str:
    """MIDI pitch number to a name with octave, for example 72 -> 'C5'."""
    return str(note.Note(midi_pitch).nameWithOctave)


def part_to_note_tokens(pitches: list[float], durations: list[float]) -> list[str]:
    """Combined pitch+duration tokens for one part.

    Rests are kept, but consecutive rests are merged, capped at MAX_REST_LENGTH, and dropped from the
    start and end of the part. Zero-length elements (grace notes) are skipped.
    """
    tokens = []
    rest = 0.0
    for e_pitch, e_duration in zip(pitches, durations):
        if e_duration <= 0:
            continue
        if e_pitch == 0:
            rest += e_duration
            continue
        if rest > 0 and len(tokens) > 0:
            tokens.append(make_note_token(REST_TOKEN, quantize_duration(min(rest, MAX_REST_LENGTH))))
        rest = 0.0
        tokens.append(make_note_token(pitch_name(e_pitch), quantize_duration(e_duration)))
    return tokens


def get_dir(composer: str, instruments: list[str]) -> str:
    """The data folder of a composer and instrument (created if needed)."""
    crt_dir = os.path.join(DATA_ROOT, composer, instruments[0] if instruments else 'all')
    os.makedirs(crt_dir, exist_ok=True)
    return crt_dir


def lcs(a: list[float], b: list[float]) -> list[float]:
    """Longest common contiguous run of a and b (used by FILTER_PARTS)."""
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
    return a[end_index - length: end_index]


def _read_cached_parts(composer: str, instruments: list[str]) -> tuple[list[list[float]], list[list[float]]]:
    """Pitches and durations of every cached part of the composer whose instrument name contains one of
    the given instrument names (every part when none is given).

    The files of each folder are shuffled with the global random generator, so the part order (and the
    train/validation split made from it) depends on the seed.
    """
    ensure_cache()
    all_pitches = []
    all_durations = []
    for root, _, files in os.walk(CACHE_ROOT):
        random.shuffle(files)
        for file_name in files:
            if not file_name.endswith('.json') or composer.lower() not in file_name.lower():
                continue
            with open(os.path.join(root, file_name), 'rt') as file:
                parts = json.load(file)
            for part in parts:
                if len(instruments) == 0:
                    all_pitches.append(parts[part][0])
                    all_durations.append(parts[part][1])
                else:
                    for instr in instruments:
                        if instr.lower() in part.lower():
                            all_pitches.append(parts[part][0])
                            all_durations.append(parts[part][1])
    return all_pitches, all_durations


def _filter_parts(all_pitches: list[list[float]], all_durations: list[list[float]]):
    """Drops parts with 25% or more rests and parts that share a run of over 75% of their length with a
    longer part already kept."""
    log('Excluding duplicate parts...')
    order = sorted(range(len(all_pitches)), key=lambda idx: -len(all_pitches[idx]))
    all_pitches = [all_pitches[idx] for idx in order]
    all_durations = [all_durations[idx] for idx in order]
    valid = [True]
    for idx in tqdm(range(1, len(all_pitches))):
        ok = len(all_pitches[idx]) > 0 and len(all_durations[idx]) > 0
        ok = ok and sum(p == 0 for p in all_pitches[idx]) / len(all_pitches[idx]) < 0.25
        if ok:
            for idy in range(idx):
                if valid[idy] and len(lcs(all_pitches[idx], all_pitches[idy])) / \
                        min(len(all_pitches[idx]), len(all_pitches[idy])) > 0.75:
                    ok = False
                    break
        valid.append(ok)
    log('Excluded', valid.count(False), 'parts!')
    return ([p for p, ok in zip(all_pitches, valid) if ok],
            [d for d, ok in zip(all_durations, valid) if ok])


def generate_input(composer: str, instruments: list[str]):
    """Writes note_input.txt, pitch_input.txt and duration_input.txt unless they already exist."""
    crt_dir = get_dir(composer, instruments)
    pth_pitches = os.path.join(crt_dir, 'pitch_input.txt')
    pth_durations = os.path.join(crt_dir, 'duration_input.txt')
    pth_notes = os.path.join(crt_dir, 'note_input.txt')
    if os.path.exists(pth_pitches) and os.path.exists(pth_durations) and os.path.exists(pth_notes):
        return

    log('Generating input data...')
    all_pitches, all_durations = _read_cached_parts(composer, instruments)
    if FILTER_PARTS:
        all_pitches, all_durations = _filter_parts(all_pitches, all_durations)

    total_units = 0
    with open(pth_pitches, 'wt') as file_pitches, open(pth_durations, 'wt') as file_durations:
        for pitches, durations in zip(all_pitches, all_durations):
            str_pitches = []
            str_durations = []
            for e_pitch, e_duration in zip(pitches, durations):
                if e_pitch == 0:
                    if FILTER_RESTS:
                        continue
                    str_pitches.append(REST_TOKEN)
                else:
                    str_pitches.append(pitch_name(e_pitch))
                str_durations.append(str(duration.Duration(e_duration).quarterLength))
            file_pitches.write(' '.join(str_pitches) + '\n')
            file_durations.write(' '.join(str_durations) + '\n')
            total_units += len(str_pitches)

    total_notes = 0
    with open(pth_notes, 'wt') as file_notes:
        for pitches, durations in zip(all_pitches, all_durations):
            tokens = part_to_note_tokens(pitches, durations)
            if len(tokens) > 0:
                file_notes.write(' '.join(tokens) + '\n')
                total_notes += len(tokens)
    log(composer.upper(), 'data has', total_units, 'units and', total_notes, 'note tokens...')


def generate_output(composer: str, instruments: list[str], kind: str = 'note') -> tuple[str, str]:
    """Writes the generated tokens as MIDI and MusicXML and returns both paths.

    For the note model it reads note_output.txt (and also splits it into pitch_output.txt and
    duration_output.txt); otherwise pitch_output.txt plus duration_output.txt (quarter notes if the
    latter is missing).
    """
    crt_dir = get_dir(composer, instruments)
    pth_note = os.path.join(crt_dir, 'note_output.txt')
    pth_pitch = os.path.join(crt_dir, 'pitch_output.txt')
    pth_duration = os.path.join(crt_dir, 'duration_output.txt')
    pth_midi = os.path.join(crt_dir, 'midi_output.mid')
    pth_xml = os.path.join(crt_dir, 'xml_output.musicxml')

    log('Generating output data...')
    if kind == 'note':
        with open(pth_note, 'rt') as file:
            tokens = [split_note_token(token) for token in file.read().split()]
        pitches = [pitch for pitch, _ in tokens]
        durations = [quarter_length for _, quarter_length in tokens]
        with open(pth_pitch, 'wt') as file:
            file.write(' '.join(pitches))
        with open(pth_duration, 'wt') as file:
            file.write(' '.join(f'{quarter_length:g}' for quarter_length in durations))
    else:
        with open(pth_pitch, 'rt') as file:
            pitches = file.read().split()
        if os.path.exists(pth_duration):
            with open(pth_duration, 'rt') as file:
                durations = [float(quarter_length) for quarter_length in file.read().split()]
        else:
            durations = [0.25 for _ in pitches]
    assert len(pitches) == len(durations)

    composition = stream.Stream()
    composition.append(clef.TrebleClef())
    composition.append(instrument.Violin())
    for pitch, quarter_length in zip(pitches, durations):
        quarter_length = quantize_duration(quarter_length)
        if pitch == REST_TOKEN:
            composition.append(note.Rest(quarterLength=quarter_length))
        else:
            composition.append(note.Note(pitch, quarterLength=quarter_length))
    composition.makeMeasures(inPlace=True)
    composition.write('midi', pth_midi)
    composition.write('musicxml', pth_xml)
    return pth_midi, pth_xml

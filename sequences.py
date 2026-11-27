"""Preparing token sequences for training: reading, splitting, transposing, vocabulary and windows.

A "sentence" is one part (one line of a ``*_input.txt`` file) as a list of tokens.
"""
import collections
import os

import numpy as np
from music21 import note, pitch

from data import REST_TOKEN, TOKEN_SEPARATOR, generate_input, get_dir

IGNORE_INDEX = -100   # target value the loss skips (padding)
OVERLAP_RUN = 12      # an extra part sharing a run of this many tokens with a reference part is left out


def read_sentences(pth: str) -> list[list[str]]:
    with open(pth, 'rt') as file:
        return [sentence.split() for sentence in file.read().strip('\n').split('\n')]


def build_vocabulary(sentences: list[list[str]]) -> dict[str, int]:
    """Token -> id, most frequent token first (ties broken alphabetically)."""
    counter = collections.Counter(word for sentence in sentences for word in sentence)
    elements = [word for word, _ in sorted(counter.items(), key=lambda x: (-x[1], x[0]))]
    assert '' not in elements
    return {word: idx for idx, word in enumerate(elements)}


def split_data(data: list[list], ratio: float) -> tuple[list[list], list[list]]:
    """Splits whole parts so that the first part holds about `ratio` of all tokens. Splitting by part,
    not by window, keeps overlapping windows from leaking across the split."""
    total = sum(len(sentence) for sentence in data)
    running = 0
    for idx, sentence in enumerate(data):
        if running >= total * ratio:
            return data[:idx], data[idx:]
        running += len(sentence)
    return data[:-1], data[-1:]


def transpose_sentences(sentences: list[list[str]], reference: list[list[str]], max_shift: int,
                        kind: str) -> list[list[str]]:
    """Each part shifted by -max_shift..+max_shift semitones (shift 0 is the part itself). A copy is kept
    only if all its pitches stay inside the pitch range of `reference`, so the instrument range holds."""
    cache: dict[str, int] = {}

    def split(word: str) -> tuple[str, str]:
        if kind == 'note':
            pitch_part, duration_part = word.rsplit(TOKEN_SEPARATOR, 1)
            return pitch_part, TOKEN_SEPARATOR + duration_part
        return word, ''

    def to_midi(name: str) -> int:
        if name not in cache:
            cache[name] = pitch.Pitch(name).midi
        return cache[name]

    midis_reference = [to_midi(p) for sentence in reference for p, _ in map(split, sentence) if p != REST_TOKEN]
    lowest, highest = min(midis_reference), max(midis_reference)
    names = {m: str(note.Note(m).nameWithOctave) for m in range(lowest, highest + 1)}
    result = []
    for sentence in sentences:
        words = [split(word) for word in sentence]
        midis = [to_midi(p) for p, _ in words if p != REST_TOKEN]
        for shift in range(-max_shift, max_shift + 1):
            if not midis or min(midis) + shift < lowest or max(midis) + shift > highest:
                continue
            if shift == 0:
                result.append(sentence)
            else:
                result.append([word if p == REST_TOKEN else names[to_midi(p) + shift] + d
                               for word, (p, d) in zip(sentence, words)])
    return result


def extra_sentences(composer: str, kind: str, reference: list[list[str]], vocabulary: list[list[str]]) -> list[list[str]]:
    """Parts of the composer's other instruments, for training only.

    A part is kept when it fits the pitch range of the reference parts, uses only tokens that
    `vocabulary` already has (so the vocabulary does not grow), and shares no OVERLAP_RUN-token run with
    a reference part (this drops the reference parts themselves and lines that double them, so nothing
    leaks into validation).
    """
    generate_input(composer, [])
    sentences = read_sentences(os.path.join(get_dir(composer, []), kind + '_input.txt'))
    known = set(word for sentence in vocabulary for word in sentence)
    seen = set(tuple(sentence[i:i + OVERLAP_RUN])
               for sentence in reference for i in range(len(sentence) - OVERLAP_RUN + 1))
    in_range = transpose_sentences([sentence for sentence in sentences if all(word in known for word in sentence)],
                                   reference, 0, kind)
    return [sentence for sentence in in_range
            if not any(tuple(sentence[i:i + OVERLAP_RUN]) in seen for i in range(len(sentence) - OVERLAP_RUN + 1))]


def generate_xy(data: list[list[int]], number_of_steps: int, window_stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Training windows. Each input is number_of_steps tokens and its target is the same window shifted by
    one, so the loss covers every timestep. Windows start every window_stride tokens, plus one aligned to
    the end of the part. Parts shorter than a window are right-padded (inputs with 0, targets with
    IGNORE_INDEX); the model is causal, so the padding never affects the real positions."""
    x = []
    y = []
    for sentence in data:
        if len(sentence) < 2:
            continue
        last_start = max(0, len(sentence) - 1 - number_of_steps)
        starts = list(range(0, last_start + 1, window_stride))
        if starts[-1] != last_start:
            starts.append(last_start)
        for start in starts:
            window = sentence[start:start + number_of_steps + 1]
            pad = number_of_steps + 1 - len(window)
            x.append(window[:-1] + [0] * pad)
            y.append(window[1:] + [IGNORE_INDEX] * pad)
    x = np.array(x, dtype=int).reshape((-1, number_of_steps))
    y = np.array(y, dtype=int).reshape((-1, number_of_steps))
    return x, y

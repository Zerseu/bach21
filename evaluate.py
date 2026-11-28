"""Measure a trained model: its loss on held-out music, and how its generated music compares with the real parts.

    python evaluate.py bach violin
    python evaluate.py bach violin --samples 48 --length 384 --temperature 0.9 --rhythm-temperature 1.0

It prints two tables.

1. Held-out loss. The per-token cross-entropy (nats) on the validation parts, the 20% of the instrument's
   parts the model never trained on. This is the same number as the best validation_loss in note_log.csv.
   For the note model it is also split into "which pitch (or rest) comes next" and "how long is it, given
   the pitch", and shown by how many tokens of context the model had.

2. Generated music against real music. `--samples` pieces of `--length` new tokens are sampled (each primed
   with real notes, exactly as generate.py does) and compared with random passages of the same length from
   the instrument's real parts. A good model has numbers close to the real column. Use `--length 1000` to
   check full-length pieces: some problems, such as a piece getting stuck in a loop, only show there.

Nothing is written to disk: the pieces are only measured. Run train.py first.
"""
import argparse
import collections
import math
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

from config import log, seed_everything
from data import REST_TOKEN, split_note_token
from model import TRAIN_RATIO, Worker
from sequences import IGNORE_INDEX, generate_xy, note_token_features, read_sentences, split_data

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)   # semitones above the key note
LOOP_DISTINCT = 0.5                    # a piece is "stuck in a loop" below this share of distinct 4-grams
REAL_PASSAGES = 200                    # random real passages the generated pieces are compared with
COPY_SEED_LENGTH = 4                   # corpus matches are searched from shared runs of this many tokens


# ----- held-out loss -----

def held_out_loss(worker: Worker) -> dict[str, float]:
    """Cross-entropy of the model on the validation windows, in nats per token.

    Returns the overall loss ('loss'), its perplexity, the number of scored targets, the loss by context
    length ('context a-b': targets at window positions a..b, which had that many tokens before them), and
    for note tokens the split into 'pitch' and 'duration' parts.

    The split uses the chain rule: -log P(token) = -log P(pitch) - log P(duration | pitch), where
    P(pitch) is the summed probability of all tokens with that pitch. The two parts add up to the loss.
    """
    c = worker.c
    sentences = read_sentences(worker.path('input.txt'))
    _, sentences_val = split_data(sentences, TRAIN_RATIO)
    data_val = [[worker.map_direct[word] for word in sentence] for sentence in sentences_val]
    window_stride = c['window_stride'] or max(1, c['number_of_steps'] // 4)
    x, y = generate_xy(data_val, c['number_of_steps'], window_stride)
    x, y = torch.from_numpy(x).long(), torch.from_numpy(y).long()

    with torch.no_grad():
        log_probabilities = torch.cat([F.log_softmax(worker.model(x[start:start + 64]), dim=-1)
                                       for start in range(0, len(x), 64)])           # (windows, T, vocabulary)
    mask = y != IGNORE_INDEX
    targets = y.clamp(min=0)   # padding targets become token 0 here and are masked out below
    nll = -log_probabilities.gather(2, targets.unsqueeze(-1)).squeeze(-1)             # (windows, T)
    count = int(mask.sum())
    result = {'loss': float(nll[mask].sum()) / count, 'targets': count}
    result['perplexity'] = math.exp(result['loss'])

    for first, last in ((0, 3), (4, 15), (16, 31), (32, 63), (64, c['number_of_steps'] - 1)):
        if first <= last:
            part = mask[:, first:last + 1]
            if int(part.sum()) > 0:
                result[f'context {first}-{last}'] = float(nll[:, first:last + 1][part].sum()) / int(part.sum())

    if worker.kind == 'note':
        # Row of each token's pitch (the rest counts as a pitch) among the distinct pitches.
        names = [split_note_token(worker.map_reverse[idx])[0] for idx in range(worker.vocabulary_size)]
        rows = {name: row for row, name in enumerate(sorted(set(names)))}
        token_pitch = torch.tensor([rows[name] for name in names])
        pitch_probability = torch.zeros(log_probabilities.shape[:2] + (len(rows),))
        pitch_probability.index_add_(2, token_pitch, log_probabilities.exp())        # sum tokens per pitch
        pitch_nll = -pitch_probability.gather(2, token_pitch[targets].unsqueeze(-1)).squeeze(-1).log()
        result['pitch'] = float(pitch_nll[mask].sum()) / count
        result['duration'] = result['loss'] - result['pitch']
    return result


# ----- statistics of a token sequence -----

def passage_statistics(tokens: list[str], midi_of: dict[str, int], pitch_range: tuple[int, int]) -> dict[str, float]:
    """Simple melodic and rhythmic measures of one passage of note tokens.

    midi_of maps a token to its MIDI pitch (-1 for rests); pitch_range is the (low, high) MIDI range
    that holds 99.8% of the real notes, used for 'outside range'.
    """
    durations = [split_note_token(token)[1] for token in tokens]
    notes = [midi_of[token] for token in tokens if midi_of[token] >= 0]
    result = {'rests': 1.0 - len(notes) / len(tokens)}

    # Pitch classes: how evenly the 12 are used (entropy), and how many notes fit the best major scale.
    classes = collections.Counter(midi % 12 for midi in notes)
    total = max(sum(classes.values()), 1)
    result['pitch-class entropy (bits)'] = -sum(n / total * math.log2(n / total) for n in classes.values())
    result['in key'] = max(sum(classes[(key + step) % 12] for step in MAJOR_SCALE) for key in range(12)) / total

    # Melodic motion between consecutive notes (rests in between are skipped).
    intervals = [abs(b - a) for a, b in zip(notes, notes[1:])]
    moves = max(len(intervals), 1)
    result['repeated pitch'] = sum(i == 0 for i in intervals) / moves
    result['steps (1-2 semitones)'] = sum(1 <= i <= 2 for i in intervals) / moves
    result['leaps over an octave'] = sum(i > 12 for i in intervals) / moves
    result['outside range'] = sum(not pitch_range[0] <= midi <= pitch_range[1] for midi in notes) / total

    # Rhythm and repetition.
    counts = collections.Counter(durations)
    result['duration entropy (bits)'] = -sum(n / len(durations) * math.log2(n / len(durations))
                                             for n in counts.values())
    result['distinct 4-grams'] = distinct_grams(tokens)
    # Stuck in a loop: in the second half of the passage, fewer than half of the 4-token runs are new.
    # This is 0 or 1 for one passage; averaged over passages it is the share of them that got stuck.
    result['stuck in a loop (share of pieces)'] = float(distinct_grams(tokens[len(tokens) // 2:]) < LOOP_DISTINCT)
    return result


def distinct_grams(tokens: list[str], size: int = 4) -> float:
    """Share of the runs of `size` consecutive tokens that are different from each other: 1.0 when no run
    occurs twice, close to 0 when the passage repeats one short figure over and over."""
    grams = [tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1)]
    return len(set(grams)) / max(len(grams), 1)


def longest_copy(tokens: list[str], corpus: list[list[str]], index: dict[tuple, list[tuple[int, int]]]) -> int:
    """Length of the longest run of `tokens` that also occurs, token for token, in a corpus part.

    index maps every run of COPY_SEED_LENGTH corpus tokens to the places (part, position) where it
    starts. Each match of such a run is extended as far as both sequences agree. Runs shorter than
    COPY_SEED_LENGTH are not searched and count as COPY_SEED_LENGTH - 1.
    """
    best = COPY_SEED_LENGTH - 1
    for start in range(len(tokens) - COPY_SEED_LENGTH + 1):
        for part, position in index.get(tuple(tokens[start:start + COPY_SEED_LENGTH]), ()):
            sentence = corpus[part]
            length = COPY_SEED_LENGTH
            while (start + length < len(tokens) and position + length < len(sentence)
                   and tokens[start + length] == sentence[position + length]):
                length += 1
            best = max(best, length)
    return best


def average(rows: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each statistic over several passages."""
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


# ----- sampling -----

def generate_pieces(worker: Worker, samples: int, length: int, seed: int) -> list[list[str]]:
    """Samples `samples` pieces of `length` new tokens, all at once as one batch.

    Each piece is primed like Worker.generate: with number_of_steps real tokens from a random place in the
    training file. The motif fallback is not applied (it is off for the note model).
    """
    seed_everything(seed)
    steps = worker.c['number_of_steps']
    with open(worker.path('input.txt'), 'rt') as file:
        content = file.read().split()
    contexts = []
    for _ in range(samples):
        start = random.randrange(0, len(content) - steps)
        contexts.append([worker.map_direct[word] for word in content[start:start + steps]])
    context = torch.tensor(contexts, dtype=torch.long)
    pieces = [[] for _ in range(samples)]
    with torch.no_grad():
        for _ in range(length):
            logits = worker.model(context[:, -steps:])[:, -1, :]
            tokens = [worker.draw(logits[row:row + 1])[1] for row in range(samples)]
            context = torch.cat([context, torch.tensor(tokens)[:, None]], dim=1)
            for piece, token in zip(pieces, tokens):
                piece.append(worker.map_reverse[token])
    return pieces


def compare_with_real(worker: Worker, samples: int, length: int, seed: int) -> tuple[dict, dict]:
    """Average statistics of real passages and of generated pieces: (real, generated)."""
    corpus = read_sentences(worker.path('input.txt'))
    words = sorted(set(word for sentence in corpus for word in sentence) | set(worker.map_direct))
    midi_of = dict(zip(words, note_token_features(words)[0]))
    real_notes = np.array([midi_of[word] for sentence in corpus for word in sentence if midi_of[word] >= 0])
    pitch_range = (int(np.percentile(real_notes, 0.1)), int(np.percentile(real_notes, 99.9)))

    # Real passages: random places in parts that are long enough, longer parts more often.
    seed_everything(seed)
    long_parts = [sentence for sentence in corpus if len(sentence) >= length]
    if not long_parts:
        raise ValueError(f'No part of the corpus has {length} tokens; use a smaller --length.')
    real = []
    for _ in range(REAL_PASSAGES):
        sentence = random.choices(long_parts, weights=[len(part) for part in long_parts])[0]
        start = random.randrange(0, len(sentence) - length + 1)
        real.append(passage_statistics(sentence[start:start + length], midi_of, pitch_range))

    index = collections.defaultdict(list)
    for part, sentence in enumerate(corpus):
        for position in range(len(sentence) - COPY_SEED_LENGTH + 1):
            index[tuple(sentence[position:position + COPY_SEED_LENGTH])].append((part, position))
    pieces = generate_pieces(worker, samples, length, seed)
    generated = average([passage_statistics(piece, midi_of, pitch_range) for piece in pieces])
    generated['longest copy from the corpus (tokens, median)'] = float(np.median(
        [longest_copy(piece, corpus, index) for piece in pieces]))
    return average(real), generated


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained bach21 model.')
    parser.add_argument('composer', help='composer the model was trained on, for example bach')
    parser.add_argument('instruments', nargs='*', help='instrument(s) the model was trained on, for example violin')
    parser.add_argument('--kind', default='note', help='config.ini section / model kind (default: note)')
    parser.add_argument('--samples', type=int, default=24, help='pieces to generate (default: 24; 0 skips table 2)')
    parser.add_argument('--length', type=int, default=384, help='new tokens per piece (default: 384)')
    parser.add_argument('--seed', type=int, default=0, help='random seed for prompts and sampling (default: 0)')
    parser.add_argument('--temperature', type=float, default=None, help='override the temperature of config.ini')
    parser.add_argument('--rhythm-temperature', type=float, default=None,
                        help='override rhythm_temperature of config.ini (0 = one temperature for whole tokens)')
    parser.add_argument('--top-p', type=float, default=None, help='override top_p of config.ini')
    arguments = parser.parse_args()

    worker = Worker(arguments.composer, arguments.instruments, arguments.kind)
    try:
        worker.load()
    except FileNotFoundError as error:
        sys.exit(str(error))
    if arguments.temperature is not None:
        worker.c['temperature'] = arguments.temperature
    if arguments.rhythm_temperature is not None:
        worker.c['rhythm_temperature'] = arguments.rhythm_temperature
    if arguments.top_p is not None:
        worker.c['top_p'] = arguments.top_p

    loss = held_out_loss(worker)
    log(f"Held-out loss: {loss['loss']:.4f} nats per token (perplexity {loss['perplexity']:.1f}) "
        f"over {loss['targets']} targets")
    if 'pitch' in loss:
        log(f"  pitch or rest: {loss['pitch']:.4f}   duration given the pitch: {loss['duration']:.4f}")
    for key, value in loss.items():
        if key.startswith('context'):
            log(f"  with {key.split()[1]:>7} tokens of context: {value:.4f}")

    if arguments.samples > 0 and worker.kind == 'note':
        real, generated = compare_with_real(worker, arguments.samples, arguments.length, arguments.seed)
        rhythm = ''
        if worker.c['rhythm_temperature'] > 0:
            rhythm = f", rhythm temperature {worker.c['rhythm_temperature']:g}"
        log(f"\nGenerated music ({arguments.samples} pieces of {arguments.length} tokens, temperature "
            f"{worker.c['temperature']:g}{rhythm}, top_p {worker.c['top_p']:g}) against {REAL_PASSAGES} real passages:")
        log(f"  {'':46s}{'real':>9s}{'generated':>11s}")
        for key, value in generated.items():
            log(f"  {key:46s}{real[key]:9.3f}{value:11.3f}" if key in real else f"  {key:46s}{'-':>9s}{value:11.3f}")
    elif arguments.samples > 0:
        log('\nThe comparison of generated and real music is only available for the note model.')


if __name__ == '__main__':
    main()

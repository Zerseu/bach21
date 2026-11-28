"""Frequent corpus motifs (repeated runs of 4 to 8 tokens) and the motif fallback used while sampling.

When ``motif_threshold`` in config.ini is above 0 and the model's top probability for the next token is
at or below it, generation continues the most frequent motif that matches the end of the sequence
instead of the sampled token. The [note] model has it off (motif_threshold = 0).
"""
import collections

MOTIF_MIN_LENGTH = 4
MOTIF_MAX_LENGTH = 8


def _is_varied(motif: tuple[int, ...]) -> bool:
    """False for motifs that repeat a single token."""
    return len(set(motif)) > 1


def query_motifs(tokens: list[int], map_reverse: dict[int, str], motif_length: int,
                 motif_filter: bool = False) -> dict[str, int]:
    """Motifs of one length that occur more than once, as 'token token ...' -> count, most frequent first."""
    # Note: the range stops one window short, so the motif ending at the last token is not counted.
    counts = collections.Counter(tuple(tokens[start:start + motif_length])
                                 for start in range(len(tokens) - motif_length))
    motifs = {' '.join(map_reverse[word] for word in motif): occurrences
              for motif, occurrences in counts.items()
              if occurrences > 1 and (not motif_filter or _is_varied(motif))}
    return dict(sorted(motifs.items(), key=lambda item: item[1], reverse=True))


def query_all_motifs(pth: str, map_direct: dict[str, int], map_reverse: dict[int, str],
                     motif_filter: bool = False) -> dict[str, int]:
    """Motifs of every length from MOTIF_MIN_LENGTH to MOTIF_MAX_LENGTH in a token file, most frequent first.
    The file's parts are joined into one stream."""
    with open(pth, 'rt') as file:
        tokens = [map_direct[word] for sentence in file.read().strip('\n').split('\n') for word in sentence.split()]
    motifs = {}
    for motif_length in range(MOTIF_MIN_LENGTH, MOTIF_MAX_LENGTH + 1):
        motifs.update(query_motifs(tokens, map_reverse, motif_length, motif_filter))
    return dict(sorted(motifs.items(), key=lambda item: item[1], reverse=True))


def motif_continuation(motifs: dict[str, int], map_direct: dict[str, int], seq: list[int]):
    """The next token of the most frequent motif whose beginning matches the end of seq, or None.

    A motif of n tokens matches when its first n - 1 tokens equal the last n - 1 tokens of seq; its last
    token is then the continuation. motifs must be ordered most frequent first (as query_all_motifs
    returns them), since the first match wins.
    """
    for motif in motifs:
        motif_ids = [map_direct[word] for word in motif.split()]
        length = min(len(seq), len(motif_ids) - 1)
        if seq[-length:] == motif_ids[-length - 1:-1]:
            return motif_ids[-1]
    return None

# Results

All numbers are for Bach's violin parts, measured on CPU in November 2026. **Held-out loss** is the
per-token cross-entropy (nats) on the 20% of violin parts held out for validation, untransposed real
music. Lower is better; it is the `validation_loss` column of `note_log.csv`, and
`python evaluate.py bach violin` prints it for a trained model.

The validation set is 14 parts with 17,824 tokens (67,137 scored targets, because the validation windows
overlap). One of the 14 parts holds almost half of those tokens, so differences of one or two hundredths
between runs should not be read as meaningful.

The short version: the current defaults predict held-out Bach better and train about six times faster
than the previous ones. Whether the music they generate is better is a mixed picture, laid out under
[Generated music](#generated-music); the clear gain there comes from the new way of sampling, not from
the new network.

## Summary

| | Previous defaults | Current defaults |
| --- | --- | --- |
| Held-out loss (nats per token) | 2.854 | **2.715** |
| ... of which pitch (or rest) | 2.037 | 1.879 |
| ... of which duration, given the pitch | 0.817 | 0.836 |
| Perplexity (`exp(loss)`) | 17.4 | 15.1 |
| Weights | 606,018 | 643,699 |
| Training batches (64 windows of 128 tokens) | 14,443 (11 epochs) | 2,202 (6 epochs) |
| Training time, one CPU thread | about 98 minutes | about 16 minutes |

Both columns were trained and measured with the same code, data split and machine, on the same training
data. "Previous defaults" is the plain network, trained on fixed windows with the reload-and-decay
schedule, as this project was before the changes described below. The times are the number of batches
multiplied by the measured time per batch (0.41 s for the plain network, 0.43 s for the current one),
because the machine was shared with other runs while they trained. Per batch the current network is 4%
slower; it needs so many fewer batches that training takes about a sixth of the time.

## Training with the default settings

| Epoch | Train loss | Validation loss | Saved |
| --- | --- | --- | --- |
| 1 | 3.650 | 3.037 | yes |
| 2 | 3.105 | 2.832 | yes |
| 3 | 2.920 | 2.745 | yes |
| 4 | 2.823 | 2.715 | yes |
| 5 | 2.770 | 2.732 | |
| 6 | 2.746 | 2.726 | |

An epoch is 367 batches. The run keeps epoch 4, at **2.715**. The learning rate follows a cosine curve
to zero, so the last epochs change little. After two epochs (734 batches) the model is already below
where the previous defaults end.

For comparison, the previous defaults on the same data split (an epoch there is 1,313 batches):

| Epoch | Train loss | Validation loss | Saved |
| --- | --- | --- | --- |
| 1 | 3.679 | 3.251 | yes |
| 2 | 3.327 | 3.064 | yes |
| 3 | 3.137 | 2.923 | yes |
| 4 | 3.006 | 2.906 | yes |
| 5 | 2.917 | 2.878 | yes |
| 6 | 2.845 | 2.868 | yes |
| 7 | 2.784 | 2.872 | |
| 8 | 2.775 | 2.855 | yes |
| 9 | 2.756 | 2.884 | |
| 10 | 2.754 | 2.860 | |
| 11 | 2.755 | 2.854 | yes |

That run was stopped by hand after epoch 11. Its own rule (stop after three epochs in a row without a new
best) had not fired yet, but its learning rate had been cut three times, to 0.000016, and the loss had
stopped moving: epoch 11 improved on epoch 8 by 0.0007.

## What each change contributes

Each row adds one change to the row above it. Every run had the same budget of 1,953 batches (16 million
token predictions, about as many as the default training run), the same seed and the same schedule (random
windows, cosine learning rate 0.002, gradient clipping 1.0), so the rows are directly comparable. S2 is
the current default configuration.

| Run | Network and data | Weights | Held-out loss | Pitch | Duration |
| --- | --- | --- | --- | --- | --- |
| S0 | the plain network (previous defaults) | 606,018 | 3.155 | 2.278 | 0.878 |
| S1 | + `structured_embedding` | 622,146 | 3.068 | 2.197 | 0.871 |
| S2 | + `pitch_relations` (the current defaults) | 643,699 | 2.709 | 1.864 | 0.844 |
| S5 | + `pattern_pointers` | 650,647 | 2.678 | 1.843 | 0.834 |
| S3 | S2 + parts of Handel and Vivaldi | 649,296 | 2.649 | | |
| S6 | S2 + parts of all seven other composers | 651,805 | 2.664 | | |

What this shows:

- **`pitch_relations` is by far the largest step** (-0.36). Almost all of it is pitch: letting the
  network read and score intervals means it no longer has to learn every melodic habit once per key.
  It also learns much faster: after 250 batches S2 is at 3.15, where the plain network needs about 1,950.
- **`structured_embedding`** gives -0.09 by sharing pitch and duration between tokens.
- **`pattern_pointers`** gives another -0.03. In the held-out parts, 26% of the intervals repeat the
  interval played 6 notes earlier and 22% of the notes repeat the previous pitch, so there is a lot to
  point at. It is **off**: the generated music did not get better (see
  [Options that lower the loss further](#options-that-lower-the-loss-further)).
- **More baroque music** gives -0.06 at the same number of batches. Handel and Vivaldi add 260,000
  tokens (551 parts) that pass the same range, vocabulary and no-doubling rules as Bach's other parts.
  It is **off** for the same reason as the pointers.
- **Music from other eras helps less.** Adding Mozart, Haydn, Beethoven, Schubert and Chopin as well
  (551,000 extra tokens in all) is still better than Bach alone, but a little worse than Handel and
  Vivaldi alone at this budget.

## Generated music

### Full-length pieces

144 pieces of 1000 new tokens (the default length of `generate.py`) per column, each primed with 128 real
tokens, compared with 200 random 1000-token passages of the real violin parts
(`python evaluate.py bach violin --samples 144 --length 1000`). "Plain network" is the previous
defaults' network; "staged" is the new sampling with separate temperatures for pitch and rhythm.

| Metric | Real Bach | Previous defaults: plain network, one temperature 0.85 | Plain network, staged (0.8 / 1.0) | Current network, one temperature 0.85 | **Current defaults:** current network, staged (0.8 / 1.0) |
| --- | --- | --- | --- | --- | --- |
| Pieces stuck in a loop | 0.0% | 7.6% | 2.8% | 19.4% | 8.3% |
| In-key share (best-fitting major scale) | 0.905 | 0.854 | 0.855 | 0.840 | 0.837 |
| Pitch-class entropy (bits) | 3.12 | 3.11 | 3.12 | 3.20 | 3.20 |
| Repeated pitch | 0.150 | 0.348 | 0.349 | 0.297 | 0.268 |
| Steps | 0.264 | 0.297 | 0.318 | 0.286 | 0.282 |
| Leaps larger than an octave | 0.025 | 0.013 | 0.007 | 0.008 | 0.007 |
| Notes outside the usual range | 0.000 | 0.020 | 0.018 | 0.012 | 0.009 |
| Share of rests | 0.203 | 0.213 | 0.225 | 0.209 | 0.206 |
| Duration entropy (bits) | 1.89 | 1.53 | 1.96 | 1.44 | 1.91 |
| Distinct 4-grams (lower = more repetition) | 0.825 | 0.779 | 0.864 | 0.732 | 0.854 |
| Longest verbatim copy from the corpus (tokens, median) | - | 7 | 7 | 7 | 6.5 |

A piece counts as stuck in a loop when fewer than half of the 4-token runs in its second half are new.
Steps are intervals of one or two semitones between consecutive notes (rests in between are skipped); the
usual range is the pitch range that holds 99.8% of the real notes.

What this shows:

- **Staged sampling is what improves the pieces, for both networks.** With one temperature the rhythm
  is monotonous (1.4 to 1.5 bits of duration entropy against 1.9 for real Bach), and a temperature low
  enough to hold the key lets pieces lock into a figure. Sampling the pitch at 0.8 and rests and durations
  at 1.0 brings the rhythm to the real level and cuts the stuck pieces to less than half.
- **The music-aware network does not generate better pieces than the plain one.** Under the same
  sampling it gets stuck more often (8.3% against 2.8%, and 19.4% against 7.6% with one temperature) and
  holds the key a little less well (0.837 against 0.855). It repeats a pitch less often (0.27 against
  0.35, real Bach 0.15) and strays outside the violin's range half as often. A network that thinks in
  intervals follows a stray note into a new key, or a figure into a loop, more readily than one that
  knows each key by heart; the loss, which is measured on real music one note at a time, does not see
  what happens when the model builds on its own output for a thousand notes.
- **Why it is the default anyway.** It predicts real music clearly better, it trains in a sixth of the
  time, and against the previous defaults as a whole the pieces are no worse: as many get stuck (8.3%
  against 7.6%), the key holds almost as well (0.837 against 0.854), and rhythm, pitch repetition and
  range are closer to real Bach. If stable pieces matter more to you than the loss and the training
  time, the plain network with staged sampling is the better choice; see
  [Going back to the plain network](#going-back-to-the-plain-network).
- **Both are still unlike real Bach:** too many repeated pitches, hardly any leaps over an octave, and
  a key that holds less well (real passages 0.905).
- Neither copies the corpus: the longest verbatim run is about 7 tokens.

With 144 pieces a share such as 8% is uncertain by about 2 percentage points either way, the in-key share
by about 0.01, and repeated pitch and steps by up to about 0.03 (estimated from the runs described under
[How noisy these statistics are](#how-noisy-these-statistics-are)). Differences smaller than that between
two columns mean nothing.

### Sampling settings

The current network at other settings (144 pieces of 1000 tokens each):

| Pitch temperature | Rhythm temperature | Stuck in a loop | In key | Repeated pitch | Duration entropy | Distinct 4-grams |
| --- | --- | --- | --- | --- | --- | --- |
| 0.85, whole tokens | - | 19.4% | 0.840 | 0.297 | 1.44 | 0.732 |
| 0.8 | 0.95 | 13.2% | 0.842 | 0.290 | 1.76 | 0.812 |
| **0.8** | **1.0** | 8.3% | 0.837 | 0.268 | 1.91 | 0.854 |
| 0.85 | 1.0 | 4.2% | 0.818 | 0.232 | 1.88 | 0.889 |

A higher temperature breaks loops and costs key stability. `temperature = 0.85` halves the stuck pieces
again; the default stops at 0.8 because the key is this network's weak side. Restricting the pitch to a
nucleus (`top_p` 0.85 to 0.9) held the key best in tests with 384-token pieces on an earlier model (0.90
to 0.91), but it removed the large leaps almost entirely and raised the repeated pitches, so it is off.

### Options that lower the loss further

Two more changes lower the held-out loss a good deal. Both are off, because the generated pieces did not
get better.

| Settings (full training run) | Batches | Held-out loss | Stuck in a loop (0.8 / 0.95) | Stuck in a loop (0.85 / 1.0) | In key (0.85 / 1.0) |
| --- | --- | --- | --- | --- | --- |
| Current defaults | 2,202 | 2.715 | 13.2% | 4.2% | 0.818 |
| + `extra_composers = handel, vivaldi` | 3,864 | 2.564 | 19.4% | 5.6% | 0.808 |
| + `extra_composers` and `pattern_pointers = true` | 3,864 | 2.540 | 19% (48 pieces) | not measured | not measured |

- **Handel and Vivaldi** (144 pieces per cell): 0.15 lower loss for 75% more training, and pieces that
  are not clearly better: the other measures go both ways within the noise, and the stuck share is a
  little higher. Its differences are within the noise one at a time, but they point the same way in
  every comparison made, including the short runs (S2 against S3 with one temperature, 48 pieces each:
  2% against 21%).
- **Pattern pointers.** In the short runs they raised the stuck share from 2% to 25% (S2 against S5, one
  temperature, 48 pieces each). In the full runs with Handel and Vivaldi the two were about the same
  (15% and 19% with pointers against 23% and 13% without, 48 pieces each). So: a slightly lower loss,
  about 14% more time per batch, no measurable benefit for the music and one clear warning sign.

### Going back to the plain network

Set these in `[note]` and train with `--retrain`. This is the previous configuration with the new
sampling; it needs about six times as long to train.

```ini
structured_embedding = false
pitch_relations = false
random_windows = false
number_of_epochs = 50
early_stopping_patience = 3
learning_rate = 0.001
lr_schedule = constant
lr_decay = 0.25
grad_clip = 0
```

### How noisy these statistics are

Single pieces differ a lot. Two runs of 48 pieces of 384 tokens that differed only in the seed were up to
0.1 apart in repeated pitch and in steps, 0.16 bits in duration entropy, 0.04 in distinct 4-grams and 0.02
in the in-key share. The share of stuck pieces came out at 2% in one run of 48 pieces and at 17% for
another model of the same configuration. Use at least a hundred pieces before believing a difference.

## Earlier experiments

The table below is from the earlier work on the plain network (data and regularisation). It was measured
on a different validation set. At that time the cache files were shuffled in file-system order, which is
not the same on every system, and the split behind this table is not the one a sorted listing gives: its
vocabulary had 813 tokens (run L has 601,965 weights), where the same settings now give 834 tokens and
606,018 weights. The files are now sorted first, so the split is the same everywhere. The validation
parts of this page are different ones, and easier ones: the previous defaults, run L, score 2.854 here
instead of 3.149. The numbers below are therefore comparable with each other but not with the tables
above.

| Run | Change | Weights | Held-out loss (old split) |
| --- | --- | --- | --- |
| baseline | hidden 256, no transposition or extra data | 1,021,892 | 3.639 |
| A | transpose training parts ±6 semitones | ~1.02M | 3.412 |
| C | tied embeddings, dropout 0.3 / 0.4, weight decay 0.1 | 988,677 | 3.395 |
| D | C without the attention layer | ~725k | 3.42 (worse) |
| G | dropout 0.4 / 0.5, weight decay 0.2, learning-rate drop on plateau | 988,677 | 3.385 |
| H | other Bach parts that fit the violin range (training only) | 988,677 | 3.272 |
| J | H, extra parts also transposed ±2 | 993,560 | 3.108 |
| E | C at hidden 192 | 594,245 | 3.427 |
| I | H at hidden 192 | 594,245 | 3.269 |
| K | J at hidden 192 | 597,912 | 3.174 |
| L | hidden 192, extra parts transposed ±4 (the previous default) | 601,965 | 3.149 |

## Findings that shaped the settings

- **Where the loss is.** With the plain network about 72% of the loss was pitch and 28% duration, which
  is why the new layers concentrate on pitch.
- **Context.** After about 32 tokens of context the loss does not fall any more (`evaluate.py` prints it
  by context length): 2.80 with 32 to 63 tokens and 2.79 with 64 to 127 for the previous defaults, 2.64
  and 2.66 for the current ones. The model makes no use of the far end of its 128-token window.
- **Metrical position.** Giving a count-based predictor the position inside the beat or bar (the running
  sum of durations) did not make the next duration easier to predict than the two previous durations
  alone (1.21 against 1.20 nats). The cache has no bar lines, drops chords and flattens tuplets, so that
  running sum drifts. A position input was therefore not added.
- **No leak into validation.** 1.1% of the validation tokens lie inside a 32-token run that also occurs
  in the (transposed) training data, and none in the extra parts.
- **Sampling.** See [Generated music](#generated-music): melody and rhythm need different temperatures,
  which is why `rhythm_temperature` exists. A sampling-time "key anchor" that favoured the pitch classes
  already used in the context was also tried and rejected: it feeds on itself and collapses the piece
  onto a few pitches (pitch-class entropy 2.7 bits at a strength that reached 0.93 in key).
- **Loss and music are different things.** Every change that lowered the held-out loss (the music-aware
  layers, the pointers, the extra composers) left the generated pieces the same or made them less
  stable. Only `evaluate.py` on full-length pieces shows that.
- **Motif fallback.** Continuing a corpus motif when the model is unsure (top probability ≤ 0.10) made
  the plain model less tonal, so `motif_threshold = 0` for `[note]`.
- **Schedule.** With the same network, random windows and the cosine schedule reach about the same loss
  per batch as fixed windows with the reload-and-decay schedule. They were adopted because a run has a
  fixed length, not because they score better.

## Ideas not tried yet

- **Stability in free generation.** The current network predicts real music better than it holds a key
  or avoids loops on its own. Training it on inputs with a few wrong notes mixed in, so that it learns to
  trust the wider context over the last few notes, is the obvious next thing to try. So is
  `structured_embedding` alone, without `pitch_relations`: in a short test it kept the key as well as the
  plain network (0.92 in 384-token pieces) with part of the loss gain, but it has not had a full run.
- **Longer memory for repeats.** In the violin parts 20% of the positions start a 16-token run that
  already occurred earlier in the same part, but only 6% of those repeats lie within 128 tokens (29%
  within 512). A model that could look that far back and copy would gain a lot, and it would give the
  generated music repeated sections.
- A larger network (hidden size 256) now that the inputs carry more information.
- Extra parts filtered for repetitiveness. Bach's other parts and the Handel and Vivaldi parts repeat
  themselves more than the violin parts do (0.74 to 0.77 distinct 4-grams in 1000 tokens against 0.83),
  so more music currently also means more accompaniment figures to imitate.
- Cleaner parts. `cache.py` flattens every part into one line of notes, so where a part holds two
  voices they end up interleaved, which shows as doubled notes and stuttering figures (the default prompt
  of seed 0 is such a passage). Real passages repeat the previous pitch 15% of the time partly for this
  reason, and the models exaggerate it. Keeping only the top voice when the cache is built would remove it.
- Parts of lower instruments moved up by an octave or two as further training data (558 more Bach
  parts, 169,000 tokens).
- A second, untouched test set. All choices on this page were made on the same 14 validation parts.

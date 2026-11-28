# Results

All numbers are for Bach's violin parts, measured on CPU in November 2026. **Held-out loss** is the
per-token cross-entropy (nats) on the 20% of violin parts held out for validation, untransposed real
music. Lower is better; it is the `validation_loss` column of `note_log.csv`.

## Training with the default settings

| Epoch | Train loss | Validation loss | Saved |
| --- | --- | --- | --- |
| 1 | 3.684 | 3.443 | yes |
| 2 | 3.321 | 3.276 | yes |
| 3 | 3.107 | 3.190 | yes |
| 4 | 2.968 | 3.163 | yes |
| 5 | 2.869 | 3.156 | yes |
| 6 | 2.789 | 3.168 | |
| 7 | 2.789 | 3.149 | yes |
| 8 | 2.767 | 3.164 | |
| 9 | 2.766 | 3.151 | |
| 10 | 2.768 | 3.149 | |

Training stops after epoch 10 (three epochs without a new best) and keeps epoch 7, at **3.149**. It
takes about 30 minutes on 4 CPU threads. The train loss stops falling after epoch 6 because each
non-improving epoch reloads the best weights and lowers the learning rate.

Before the data and regularisation work, the same model overfitted after 4 epochs: the validation loss
was 3.639 at its best and then rose every epoch while the train loss kept falling.

## How the defaults were chosen

Each run adds to the one before it unless noted.

| Run | Change | Weights | Held-out loss |
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
| **L** | hidden 192, extra parts transposed ±4 (**the default**) | 601,965 | **3.149** |

Run J has the lowest loss but is 65% larger; L was chosen as the best balance of quality and training
time.

## Generated music

Averaged over 24 samples of 384 new tokens each, compared with random 384-token passages of real violin
parts:

| Metric | Real Bach | Baseline (temp 1.0, motifs on) | Baseline (temp 1.0, motifs off) | Default (temp 0.85, motifs off) |
| --- | --- | --- | --- | --- |
| Share of rests | 0.226 | 0.283* | 0.264 | 0.258 |
| Pitch-class entropy (bits) | 3.08 | 3.19* | 3.16 | 3.02 |
| In-key share (best-fitting major scale) | 0.907 | 0.880* | 0.892 | 0.890 |
| Leaps larger than an octave | 0.026 | 0.034* | 0.045 | 0.013 |
| Distinct 4-grams (lower = more repetition) | 0.897 | 0.994* | 0.993 | 0.851 |
| Stepwise motion | 0.384 | 0.413* | 0.405 | 0.548 |
| Longest verbatim copy from the corpus (tokens, median) | - | 6* | 5 | 6 |

\* 12 samples.

What this shows:

- The default model stays in key as well as the baseline and close to real Bach, with fewer wild leaps
  and more stepwise motion than real Bach.
- It repeats itself a little more than real music (fewer distinct 4-grams), a common effect of sampling
  below temperature 1.
- It does not copy the corpus: the longest verbatim run is about 6 tokens, the same as the baseline.

## Findings that shaped the settings

- **Motif fallback.** Continuing a corpus motif when the model is unsure (top probability ≤ 0.10) made
  every model less tonal, so `motif_threshold = 0` for `[note]`. Set it to 0.10 to bring it back.
- **Temperature.** Models trained with transposed and extra data drift out of key more at temperature
  1.0 (in-key share 0.83). Temperature 0.85 brings it back to the baseline level.
- **Attention.** Removing the attention layer raised the loss from 3.395 to 3.42.
- **Rests in the default piece.** Seed 0 picks a prompt from a sparse passage with 39% rests, which is
  why the piece from the default seed has more rests than most. Other seeds do not.

## Ideas not tried yet

- Hidden size 256 with the extra parts transposed ±4 or ±6.
- Pre-training on all of Bach and fine-tuning on the violin parts with `base_model`.

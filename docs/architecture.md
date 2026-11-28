# Architecture

This page follows one run from the corpus to a finished piece. File and function names refer to the
Python modules in the project folder.

```
music21 corpus ──cache.py──▶ bach21cache/*.json ──data.py──▶ note_input.txt
                                                               │
                                         sequences.py: split, transpose, extra parts, random windows
                                                               │
                                                               ▼
                                      model.py + network.py: train ──▶ note_model.torch
                                                                            │
                                      model.py: prime with 128 real tokens, sample one at a time
                                                                            │
                                                                            ▼
                                       note_output.txt ──data.py──▶ midi_output.mid, xml_output.musicxml

                                      evaluate.py: held-out loss, generated against real music
```

## 1. Corpus cache (`cache.py`)

`rebuild_cache` parses every score in music21's built-in corpus (and, with `--external DIR`, a folder of
your own scores) and splits it by instrument with `instrument.partitionByInstrument`. For each part
`unravel_part` keeps the notes and rests in order:

- pitch: the MIDI pitch number (`pitch.ps`, so 60 is middle C), or 0 for a rest;
- duration: the length in quarter notes, ignoring tuplet scaling (`quarterLengthNoTuplets`).

Chords, and parts without a named instrument, are left out. All parts of one instrument in one score are
joined. The result is one JSON file per score, named `<composer>_<NNNN>.json`, mapping each lower-case
instrument name to `[pitches, durations]`. The composer is the first folder of the score's corpus path.

The repository ships the cache as `bach21cache.zip` (about 2,000 files); `ensure_cache` unpacks it the
first time anything needs it.

## 2. Tokens (`data.py`)

`generate_input(composer, instruments)` collects every cached part of the composer whose instrument name
contains one of the requested names, in a seeded random order (the cache files are sorted by name before
they are shuffled, so the order is the same on every operating system), and writes three files to
`bach21data/<composer>/<instrument>/`:

- `note_input.txt`, used by the default model. Each note becomes `<pitch>_<duration>`, for example
  `C5_0.5` (C5 for an eighth note) or `F#4_1.5`. Rests become `RST_<duration>`. `part_to_note_tokens`
  rounds every duration to a multiple of a 32nd note (`DURATION_GRID = 0.125`, never below one step),
  merges consecutive rests into one, caps a rest at a whole note (`MAX_REST_LENGTH = 4`), drops rests at
  the start and end of a part, and skips zero-length elements such as grace notes. Pitch names use
  music21's default spelling (C#, E-, F#, G#, B-).
- `pitch_input.txt` and `duration_input.txt`, the same parts as separate streams without rests
  (`FILTER_RESTS`). They are used by the experimental `pitch` and `duration` models and by `entropy.py`.

One line is one part. The files are built once and reused; delete them to rebuild. `FILTER_PARTS`
(off) can additionally drop parts with 25% or more rests and near-duplicate parts.

For Bach violin this gives 72 parts and about 95,000 note tokens.

## 3. Training data (`sequences.py`, `Worker.training_sentences`)

1. **Split.** `split_data` puts whole parts, in file order, into training until they hold 80% of the
   tokens (`TRAIN_RATIO`); the rest is validation. Splitting by part keeps overlapping windows of one part
   from landing on both sides.
2. **Transposition.** `transpose_sentences` adds copies of every training part shifted by
   -`transpose`..+`transpose` semitones (6 by default). A copy is kept only when all its pitches stay
   within the lowest and highest pitch of the instrument's real parts, so the instrument's range holds.
3. **Extra parts.** With `extra_training_data`, `extra_sentences` builds the composer's `all` folder and
   adds parts of other instruments that fit the same pitch range, use only tokens the instrument's data
   already has, and share no run of 12 tokens (`OVERLAP_RUN`) with an instrument part. The last rule drops
   the instrument's own parts, which are in `all` too, and lines that double them, so nothing from
   validation leaks into training. The same can be done for other composers by listing them in
   `extra_composers` (empty by default, see [results.md](results.md) for why): their parts go through
   exactly the same three rules. All extra parts are transposed by up to `transpose_extra` (4).
4. **Vocabulary.** `build_vocabulary` numbers every distinct token of training and validation, most
   frequent first. It is saved with the model as `note_vocabulary.json` (a JSON list; the index is the
   token id). With the default settings the Bach violin vocabulary has about 830 tokens.
5. **Windows.** A training example is a window of `number_of_steps` (128) input tokens; the targets are
   the same tokens shifted by one, so every position predicts the next token. A part shorter than one
   window is padded on the right (input id 0, target -100, which the loss skips).
   - *Training* (`RandomWindows`, with `random_windows = true`): every batch draws new windows. A part
     is picked with probability proportional to its length and the window starts at a random token of
     it. One epoch is `tokens / (128 * batch_size)` batches, about one visit per training token.
   - *Validation* (`generate_xy`): fixed windows that start every `window_stride` (32) tokens, plus one
     aligned to the end of the part, always the same ones. With `random_windows = false` training uses
     fixed windows of this kind too.

Validation always uses the real, untransposed parts, so its loss measures how well the model predicts
real music of that instrument.

## 4. Network (`network.py`)

```
token ids (B, T)
  → token vectors (H)                      shared with the output layer when tie_weights = true
      = token part + pitch part + duration part            (structured_embedding)
      + pitch-class part, + interval from the previous note (pitch_relations)
  → Dropout (dropout = 0.4)
  → LSTM, one layer (H)
  → causal multi-head self-attention (4 heads) + residual + LayerNorm
  → Dropout (output_dropout = 0.5)
  → scores per token (B, T, vocabulary)
      = state · token vector + bias
      + score of the interval from the last note to the token's pitch   (pitch_relations)
      [+ scores for repeating a recent pitch, interval or duration      (pattern_pointers, off)]
```

B is the batch size, T the window length and H = `hidden_size` (192). The attention layer lets each
position look directly at every earlier token in the window (a mask hides later ones), which complements
the LSTM's running memory.

A plain embedding treats the roughly 800 tokens as unrelated symbols: the network has to find out from
examples alone that `C5_0.25` and `C5_0.5` are the same pitch, that `D5` lies a step above `C5`, and it
has to learn every melodic habit again in every key. Two options, both on for `[note]`, build that
knowledge into the first and last layer, and a third, experimental one is off. They read the pitch and
the duration out of each token (`sequences.note_token_features`), so they only exist for the note model.

- **`structured_embedding`.** A token's vector is the sum of three learned parts: one of its own, one for
  its pitch (shared by all tokens with that pitch) and one for its duration (shared by all tokens with
  that duration). The output layer is tied to the same sum. A token that is rare in the data then still
  has a well-trained pitch part and duration part.
- **`pitch_relations`.** Three things. The token vector gets a fourth part for the pitch class (all C's
  share it). The input gets an embedding of the interval, in semitones, from the previous note (clipped
  to two octaves, `MAX_INTERVAL`). And the output gets 49 extra scores, one per interval from -24 to +24:
  the score for "k semitones from the last note" is added to every token whose pitch lies at that
  interval. "Step down after a leap up" is then one thing to learn rather than one per key.
- **`pattern_pointers`** (off). For k = 1 to 12 (`POINTER_SPAN`) the output gets a score for "the pitch
  from k notes ago again", one for "the interval from k notes ago again" and one for "the duration from k
  tokens ago again". Each is added to all tokens that would do exactly that. Repeating figures such as
  arpeggios and rhythms that return every few notes are everywhere in this music, and the pointers do
  make the network predict real music a little better. The generated music did not get better, though,
  and in one comparison far more pieces got stuck in a loop: in free generation the network's own
  repeats are evidence for repeating again (see [results.md](results.md)). The option is kept for
  experiments.

None of these look ahead: they only use tokens at or before the current position, so the model stays
causal. The lookup tables behind them are rebuilt from the vocabulary when a model is loaded and are not
stored in the model file.

With the `[note]` settings the network has 643,699 weights for Bach violin: about 161k in the token
table and output bias, 296k in the LSTM, 149k in the attention layer, and 38k in the two music-aware
options (16k and 22k). `pattern_pointers` would add 7k.

The attribute names of `TorchModule` are the keys of the saved state, so renaming one breaks loading
existing model files. The vocabulary is not part of the state; it is saved next to it. With the three
options off the network is exactly the one from before they existed, and loads models trained then.

## 5. Training (`model.py`)

`main_train` seeds everything with `TRAIN_SEED = 0`, builds the token files if needed and calls
`Worker.train`, which skips training when a model already exists (unless `--retrain`).

- **Loss.** Cross-entropy over every non-padding position of every window. The reported train and
  validation losses are averages per token, in nats (natural log).
- **Optimiser.** AdamW with `weight_decay` 0.2, batches of 64 random windows, and the gradient norm
  clipped to `grad_clip` (1.0).
- **Learning rate** (`make_scheduler`, `lr_schedule = cosine`). It rises linearly from 0 to
  `learning_rate` (0.002) over the first `lr_warmup_steps` (100) batches and then falls along a cosine
  curve to 0 at the end of the last epoch. Large steps early find a good region quickly, the long
  decline settles into it, and a run always takes `number_of_epochs` (6) epochs.
- **Checkpointing.** After each epoch the model is evaluated on the validation windows (with dropout
  off). If the validation loss improved, the weights are saved to `note_model_best.torch`.
- **The older schedule** is still available with `lr_schedule = constant`: a fixed learning rate; after
  an epoch without improvement the best weights are loaded back and the learning rate is multiplied by
  `lr_decay`; after `early_stopping_patience` such epochs in a row training stops.
- **Output.** `note_log.csv` gets one line per epoch (`epoch,train_loss,validation_loss,best`). At the
  end the vocabulary is written and the best weights are renamed to `note_model.torch`.

With LoRA (`lora_enable`), the network is first created without adapters, `base_model` is loaded into it,
and then `apply_lora` wraps the chosen weights (`lora_targets`: LSTM input `ih` and recurrent `hh`
weights, the output layer `out`, and an extra identity projection after the embedding `emb`) as
`W + (alpha / r) * B @ dropout(A)`, with `lora_dropout` active only while training. `lora_peft_only` freezes everything but the adapters and biases.

## 6. Generation (`model.py`, `data.py`)

`main_generate` creates a `Worker`, which loads the model and its vocabulary on the CPU in evaluation
mode, then:

1. Seeds everything with `--seed`.
2. Picks a random start in `note_input.txt` (all parts joined into one stream) and takes 128 consecutive
   real tokens as the prompt.
3. Repeats `--length` times: run the model on the last 128 tokens, take its probabilities for the next
   token, and draw one (`Worker.draw`). With `rhythm_temperature` above 0 (1.0 for `[note]`) this
   happens in three steps (`sample_staged`), which only regroup the model's own probabilities:
   1. rest or note, from the summed probability of the rest tokens, at `rhythm_temperature`;
   2. for a note, the pitch, from the summed probability of all tokens with each pitch, at `temperature`
      (0.8), optionally restricted to the smallest set of pitches whose probabilities add up to `top_p`;
   3. the duration, among the tokens of the chosen pitch, at `rhythm_temperature`.

   A temperature T turns a probability p into `p^(1/T)` (renormalised). Melody and rhythm get separate
   temperatures because they need different ones: a value low enough to keep the melody in key makes the
   rhythm monotonous. With `rhythm_temperature = 0` there is one step instead: the logits are divided by
   `temperature`, optionally cut to the `top_p` nucleus of tokens, and a whole token is drawn (`sample`).
4. If `motif_threshold` > 0 and the most likely token's probability is at or below it, the next token
   of the most frequent matching corpus motif (a repeated run of 4 to 8 tokens, from `motifs.py`) is used
   instead. This is off for the `[note]` model because it made the music less tonal. The motifs are
   counted once and cached in `note_motifs.json`; delete it to count them again.
5. Writes the new tokens (and the prompt with `--keep-prompt`) to `note_output.txt`.

`generate_output` then splits each token back into pitch and duration and writes a music21 stream (violin,
treble clef, bars of 4/4) as `midi_output.mid` and `xml_output.musicxml`.

## 7. Evaluation (`evaluate.py`)

`python evaluate.py bach violin` loads the trained model and prints two things.

- **Held-out loss.** The cross-entropy on the validation windows, the same number as the best
  `validation_loss` in `note_log.csv`. It is also split into a pitch part and a duration part with the
  chain rule, `-log P(token) = -log P(pitch) - log P(duration | pitch)`, where `P(pitch)` is the summed
  probability of all tokens with that pitch (the rest counts as a pitch), and shown by how many tokens
  of context the model had inside the window.
- **Generated against real music.** It samples `--samples` (24) pieces of `--length` (384) tokens, each
  primed like `generate.py` does, and compares their averages with 200 random passages of the same
  length from the instrument's real parts: share of rests, pitch-class entropy, share of notes in the
  best-fitting major scale, repeated pitches, steps, leaps over an octave, notes outside the usual range,
  duration entropy, distinct 4-grams, the share of pieces that got stuck in a loop (fewer than half of the
  4-token runs in their second half are new), and the longest run copied verbatim from the corpus. Run
  it with `--length 1000` as well: loops only show in full-length pieces.
  `--temperature` and `--top-p` override the values in `config.ini` for one run, which makes it easy to
  see what a sampling setting does before changing it.

## Experimental model kinds

`--kind pitch` and `--kind duration` train models on `pitch_input.txt` or `duration_input.txt` with the
`[pitch]` and `[duration]` settings. They predict only pitches or only durations; `generate_output` can
combine a `pitch_output.txt` with a `duration_output.txt` (or uses sixteenth notes when there is none).
They come from earlier experiments and are not tuned; the `note` model replaces them.

## Entropy analysis

`entropy.py` reads `pitch_input.txt` and, for every part longer than 256 notes, measures its transition
entropy: for each pitch, the entropy in bits of which pitch follows it, averaged over all pitches. It
compares that with uniformly random sequences of the same length over the same set of pitches, and
saves:

- `entropy_plot_measured.png`: real against noise entropy for each part, by length;
- `entropy_plot_interpolated.png`: the entropy interpolated over part length, marked at 1000 notes.

It prints the expected entropy at 1000 notes and the noise entropy at that length. If the folder also has
a `results.csv` from the paper's experiment runs (motif threshold, temperature, expected and noise
entropy, then 10 entropies without and 10 with motifs per row), it also plots entropy against sampling
temperature for motif thresholds 0.10 and 0.25. Nothing in this repository writes `results.csv`.

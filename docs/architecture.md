# Architecture

This page follows one run from the corpus to a finished piece. File and function names refer to the
Python modules in the project folder.

```
music21 corpus ──cache.py──▶ bach21cache/*.json ──data.py──▶ note_input.txt
                                                               │
                                              sequences.py: split, transpose, extra parts, windows
                                                               │
                                                               ▼
                                      model.py + network.py: train ──▶ note_model.torch
                                                                            │
                                      model.py: prime with 128 real tokens, sample one at a time
                                                                            │
                                                                            ▼
                                       note_output.txt ──data.py──▶ midi_output.mid, xml_output.musicxml
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
contains one of the requested names, in a seeded random order, and writes three files to
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
   validation leaks into training. These parts are transposed by up to `transpose_extra` (4).
4. **Vocabulary.** `build_vocabulary` numbers every distinct token of training and validation, most
   frequent first. It is saved with the model as `note_vocabulary.json` (a JSON list; the index is the
   token id). With the default settings the Bach violin vocabulary has about 800 tokens.
5. **Windows.** `generate_xy` cuts each part into windows of `number_of_steps` (128) input tokens; the
   targets are the same tokens shifted by one, so every position predicts the next token. Windows start
   every `window_stride` (32) tokens, plus one aligned to the end of the part. A part shorter than one
   window is padded on the right (input id 0, target -100, which the loss skips).

Validation always uses the real, untransposed parts, so its loss measures how well the model predicts
real music of that instrument.

## 4. Network (`network.py`)

```
token ids (B, T)
  → Embedding (H)                          shared with the output layer when tie_weights = true
  → Dropout (dropout = 0.4)
  → LSTM, one layer (H)
  → causal multi-head self-attention (4 heads) + residual + LayerNorm
  → Dropout (output_dropout = 0.5)
  → Linear → logits (B, T, vocabulary)
```

B is the batch size, T the window length and H = `hidden_size` (192). The attention layer lets each
position look directly at every earlier token in the window (a mask hides later ones), which complements
the LSTM's running memory; removing it made the loss worse (see [results.md](results.md)).

With tied weights the network has `193 * vocabulary + 445,056` weights for H = 192: about 157k in the
embedding, 296k in the LSTM, 148k in the attention layer. That is about 602k for Bach violin.

The attribute names of `TorchModule` are the keys of the saved state, so renaming one breaks loading
existing model files. The vocabulary is not part of the state; it is saved next to it.

## 5. Training (`model.py`)

`main_train` seeds everything with `TRAIN_SEED = 0`, builds the token files if needed and calls
`Worker.train`, which skips training when a model already exists (unless `--retrain`).

- **Loss.** Cross-entropy over every non-padding position of every window. The reported train and
  validation losses are averages per token, in nats (natural log).
- **Optimiser.** AdamW with `learning_rate` 0.001 and `weight_decay` 0.2, batches of 64 shuffled windows.
  `grad_clip` can clip the gradient norm (off by default).
- **Checkpointing.** After each epoch the model is evaluated on the validation windows (with dropout
  off). If the validation loss improved, the weights are saved to `note_model_best.torch`.
- **Learning-rate drop.** If it did not improve, the best weights are loaded back and the learning rate
  is multiplied by `lr_decay` (0.25). This undoes an epoch that overfitted and continues with smaller
  steps.
- **Early stopping.** After `early_stopping_patience` (3) epochs in a row without improvement, training
  stops. Otherwise it runs `number_of_epochs` (50).
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
3. Repeats `--length` times: run the model on the last 128 tokens, take the logits after the last one,
   divide them by `temperature` (0.85), turn them into probabilities, optionally keep only the smallest
   set of tokens whose probabilities add up to `top_p`, and draw the next token.
4. If `motif_threshold` > 0 and the most likely token's probability is at or below it, the next token
   of the most frequent matching corpus motif (a repeated run of 4 to 8 tokens, from `motifs.py`) is used
   instead. This is off for the `[note]` model because it made the music less tonal. The motifs are
   counted once and cached in `note_motifs.json`; delete it to count them again.
5. Writes the new tokens (and the prompt with `--keep-prompt`) to `note_output.txt`.

`generate_output` then splits each token back into pitch and duration and writes a music21 stream (violin,
treble clef, bars of 4/4) as `midi_output.mid` and `xml_output.musicxml`.

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

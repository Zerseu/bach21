# bach21

bach21 teaches a recurrent neural network to compose classical music. It learns from the scores in
[music21](https://web.mit.edu/music21/)'s corpus, for example every violin part Bach wrote, and then
writes new music in that style as MIDI and MusicXML.

The default setup trains on Bach's violin parts and writes a 1000-note violin piece. Each note the
model writes is one token that carries both pitch and duration, so the output has real rhythm.

More detail lives in [`docs/`](docs):

- [docs/architecture.md](docs/architecture.md): the data pipeline, the network, training and sampling.
- [docs/configuration.md](docs/configuration.md): every setting in `config.ini`.
- [docs/results.md](docs/results.md): measured losses and sampling statistics, and how the defaults were chosen.

To check a trained model yourself, run `python evaluate.py bach violin`: it prints the loss on held-out
music and compares generated pieces with real ones.

## Contents

- [Requirements](#requirements)
- [Quick start on Windows](#quick-start-on-windows)
- [Running from Python directly (any OS)](#running-from-python-directly-any-os)
- [Command reference](#command-reference)
- [How it works](#how-it-works)
- [Files](#files)
- [Settings](#settings)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [License](#license)

## Requirements

- Python 3.11 or newer.
- About 1 GB of disk space for the virtual environment (mostly PyTorch) and about 80 MB for the
  unpacked corpus cache.
- A CPU is enough. With the default settings training takes roughly 6 minutes on 4 CPU cores. An
  NVIDIA GPU is used automatically when a CUDA build of PyTorch is installed.

The Python packages are pinned in [`requirements.txt`](requirements.txt): music21, PyTorch, NumPy and tqdm
for training and generation, plus matplotlib, pandas and SciPy for `entropy.py`.

## Quick start on Windows

You need [Python](https://www.python.org/downloads/) 3.11 or newer. Then, in the project folder:

1. Double-click **`train_bach_violin.bat`**, or run it from a terminal. It trains the model on Bach's
   violin parts. With the default settings this takes roughly 6 minutes on a 4-core CPU.
2. Double-click **`generate_bach_violin.bat`**. It writes a new 1000-note piece to
   `bach21data\bach\violin\midi_output.mid` and `xml_output.musicxml`.

The first time either script runs, it creates a virtual environment in your user folder, at
`%USERPROFILE%\.venvs\bach21` (for example `C:\Users\<you>\.venvs\bach21`), and installs everything from
`requirements.txt` into it. The project folder stays clean and nothing goes into your system Python. Later
runs reuse it and only reinstall when `requirements.txt` changes. To start over, delete that folder.

Both scripts pass extra options on to Python:

```bat
train_bach_violin.bat --retrain              rem train again although a model exists
generate_bach_violin.bat --seed 7            rem another piece (the same seed gives the same piece)
generate_bach_violin.bat --length 500        rem a shorter piece
generate_bach_violin.bat --keep-prompt       rem also write the real Bach notes the model was primed with
```

When a script is started by double-clicking, its window stays open at the end so you can read the
messages. It exits with code 0 on success and 1 on failure, so it can also be called from other scripts.

### Environment variables

| Variable | Effect |
| --- | --- |
| `BACH21_VENV_DIR` | Folder for the virtual environment instead of `%USERPROFILE%\.venvs\bach21`. |
| `BACH21_TORCH_INDEX_URL` | Installs PyTorch from this package index, for example `https://download.pytorch.org/whl/cu128` for an NVIDIA GPU build. The default from PyPI is the CPU build. |

Set them in the same terminal before running a script, for example
`set BACH21_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128`. `BACH21_TORCH_INDEX_URL` only matters
when packages are installed, so to switch an existing environment to the GPU build, delete the environment
folder and run a script again with it set.

### What `scripts\setup_venv.bat` does

1. Looks for Python 3.11+ through the `py` launcher (3.13, 3.12, 3.11, 3.14, then any 3.x), and then
   `python` on `PATH`.
2. Creates the virtual environment if `python.exe` is not in it yet.
3. Installs `requirements.txt` when the environment has never been set up, when `requirements.txt`
   differs from the copy saved after the last install (`requirements.installed` in the environment
   folder), or when one of the packages fails to import.
4. Sets `VENV_PYTHON` for the calling script.

## Running from Python directly (any OS)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip install -r requirements.txt
.venv/bin/python train.py bach violin
.venv/bin/python generate.py bach violin --length 1000
.venv/bin/python evaluate.py bach violin       # optional: measure the trained model
```

`train.py`, `generate.py` and `evaluate.py` take `COMPOSER [INSTRUMENT ...]` and work for any composer and
instrument in the corpus cache (`python train.py bach cello`, `python train.py mozart violin`, ...). Without
an instrument they use every part of the composer. Each composer and instrument gets its own folder in
`bach21data/`, so models for different instruments do not overwrite each other.

## Command reference

All commands run from any folder; paths are resolved relative to the project folder.

### `train.py`: train a model

```
python train.py COMPOSER [INSTRUMENT ...] [--retrain] [--kind KIND]
```

| Argument | Meaning |
| --- | --- |
| `COMPOSER` | Composer name as it appears in the cache file names, for example `bach`. |
| `INSTRUMENT` | One or more instrument names. A part is used when its instrument name contains one of them, so `violin` also matches `violin i`. Leave out to use every part. The first instrument names the data folder. |
| `--retrain` | Train again even if `note_model.torch` exists. Without it an existing model is kept and nothing is trained. |
| `--kind` | `config.ini` section and file prefix, default `note`. `pitch` and `duration` train separate experimental models. |

Training builds the token files on first use, logs the loss of every epoch, and saves the model with the
lowest validation loss. A run always has `number_of_epochs` epochs (6 by default). The model file is
only written at the end, so stopping a run with Ctrl+C keeps the previous model.

### `generate.py`: compose a piece

```
python generate.py COMPOSER [INSTRUMENT ...] [--length N] [--seed S] [--keep-prompt] [--kind KIND]
```

| Argument | Meaning |
| --- | --- |
| `--length` | Number of new tokens (notes and rests) to write, default 1000. |
| `--seed` | Random seed for the choice of prompt and for sampling, default 0. The same seed and model give the same piece. |
| `--keep-prompt` | Also write the 128 real corpus tokens the model was primed with, before the new ones. |
| `--kind` | As for `train.py`. |

It writes `note_output.txt`, `midi_output.mid` and `xml_output.musicxml` (plus `pitch_output.txt` and
`duration_output.txt`, the same piece split into pitches and durations) and prints their paths. A later
run overwrites them, so copy a piece you want to keep.

### `evaluate.py`: measure a trained model

```
python evaluate.py COMPOSER [INSTRUMENT ...] [--samples N] [--length N] [--seed S]
                   [--temperature T] [--rhythm-temperature R] [--top-p P] [--kind KIND]
```

| Argument | Meaning |
| --- | --- |
| `--samples` | Number of pieces to generate for the comparison with real music, default 24. `0` only prints the loss. |
| `--length` | New tokens per piece, default 384. Use 1000 to measure full-length pieces. |
| `--seed` | Random seed for the prompts and for sampling, default 0. |
| `--temperature`, `--rhythm-temperature`, `--top-p` | Use these sampling settings instead of the ones in `config.ini`, for this run only. |
| `--kind` | As for `train.py`. |

It prints the model's loss on the held-out parts (the same number as the best `validation_loss` of
training, split into pitch and duration) and a table that compares generated pieces with real passages:
share of rests, how well they stay in key, steps and leaps, repetition, pieces stuck in a loop, and
whether anything is copied from the corpus. The table is noisy with the default 24 pieces; use
`--samples 144` or so to compare two settings. No piece is saved. See
[docs/architecture.md](docs/architecture.md#7-evaluation-evaluatepy) for what each number means.

### Other commands

| Command | What it does |
| --- | --- |
| `python model.py bach violin` | Trains (unless a model exists) and then generates 1000 notes with seed 0. |
| `python cache.py [--external DIR] [--no-internal]` | Rebuilds `bach21cache/` from music21's corpus and, optionally, a folder of your own MIDI / MusicXML files with one sub-folder per composer. |
| `python entropy.py bach violin` | Measures the pitch transition entropy of the corpus parts and saves plots (see [docs/architecture.md](docs/architecture.md#entropy-analysis)). |

Every script accepts `--help` except `model.py` and `entropy.py`, which only take the composer and
instruments.

## How it works

1. **Corpus cache** (`cache.py`). Every score is reduced to the pitches and durations of each instrument
   and stored as JSON in `bach21cache/`. A ready-made cache ships as `bach21cache.zip` and is unpacked
   automatically.
2. **Tokens** (`data.py`). Each note becomes one token that combines pitch and duration, such as `C5_0.5`
   (C5, an eighth note) or `RST_1` (a quarter rest). Durations are rounded to 32nd notes, rests are merged
   and capped at a whole note. Each part becomes one line of `bach21data/<composer>/<instrument>/note_input.txt`.
3. **Training data** (`sequences.py`). The parts are split 80/20 into training and validation, by whole
   part. Training parts are also used transposed by up to 6 semitones. The composer's other parts that
   fit the instrument's range are added (transposed by up to 4), as long as they do not double a violin
   part. Validation is always the real, untransposed violin parts. Training uses windows of 128 tokens
   that start at random places.
4. **Model** (`network.py`). Embedding, one LSTM layer, a causal self-attention layer and an output layer
   that shares its weights with the embedding (about 640k weights with the default settings). It
   predicts the next token at every position. The first and last layer know what a token means
   musically: tokens share what they have in common (pitch, duration, pitch class), and the network
   sees and scores intervals between notes, so what it learns holds in every key.
5. **Training** (`model.py`). AdamW with weight decay and dropout for 6 epochs. The learning rate
   warms up briefly and then falls to zero along a cosine curve. After every epoch the model is tested
   on the validation parts, and the one with the lowest validation loss is kept.
6. **Generation** (`model.py`, `generate.py`). The model is primed with 128 real notes from a random place
   in the corpus and then samples one token at a time: first whether it is a rest or a note, then the
   pitch (temperature 0.8), then the duration (rhythm temperature 1.0). Melody and rhythm have separate
   temperatures because the melody needs the lower one to stay in key and the rhythm the higher one to
   stay varied. Only the new notes are written, unless `--keep-prompt` is given.

[docs/architecture.md](docs/architecture.md) explains each step in detail.

## Files

| File | What it does |
| --- | --- |
| `train_bach_violin.bat`, `generate_bach_violin.bat` | Windows entry points (see above) |
| `scripts/setup_venv.bat` | Creates the venv in `%USERPROFILE%\.venvs\bach21` and installs the dependencies; used by both `.bat` files |
| `train.py`, `generate.py` | Command-line training and generation |
| `evaluate.py` | Held-out loss of a trained model and a comparison of generated with real music |
| `config.ini` | Model, training and sampling settings, one section per model kind; `[note]` is the one used |
| `config.py` | Reads `config.ini` (every key and its default is listed in `DEFAULTS`), project paths, logging, seeding |
| `cache.py` | Builds or unpacks the corpus cache |
| `data.py` | Cache to token files, and generated tokens to MIDI / MusicXML |
| `sequences.py` | Train/validation split, transposition, extra parts, vocabulary, training windows, pitch and duration of each token |
| `network.py` | The PyTorch network, with its music-aware embedding and output layers |
| `model.py` | Training loop, sampling, and the `Worker` class that ties one model's files together |
| `motifs.py` | Frequent corpus motifs and the optional motif fallback while sampling (off for `[note]`) |
| `lora.py` | LoRA adapters for fine-tuning a trained model (off by default) |
| `entropy.py` | Entropy analysis of the corpus pitch sequences: `python entropy.py bach violin` |
| `bach21cache.zip` | The prebuilt corpus cache, unpacked to `bach21cache/` on first use |
| `docs/` | Architecture, configuration and results documentation |

Everything a run produces goes to `bach21data/<composer>/<instrument>/` (`all` instead of the instrument
when none is given):

| File | Contents |
| --- | --- |
| `note_input.txt`, `pitch_input.txt`, `duration_input.txt` | Training tokens, one part per line |
| `note_model.torch`, `note_vocabulary.json` | The trained model and its token list (index = token id) |
| `note_model_best.torch` | Best weights so far, only present while training runs |
| `note_log.csv` | Training and validation loss per epoch (`best` = 1 where the model was saved) |
| `note_motifs.json` | Cached corpus motifs, only built when `motif_threshold` > 0 |
| `note_output.txt` | The generated tokens |
| `pitch_output.txt`, `duration_output.txt` | The same piece as separate pitch and duration lists |
| `midi_output.mid`, `xml_output.musicxml` | The generated piece |
| `entropy_plot_*.png` | Plots written by `entropy.py` |

Training with `extra_training_data` also builds the token files of all parts of the composer (and of each
composer listed in `extra_composers`) in `bach21data/<composer>/all/`.

A log of every run is appended to `bach21.log` in the project folder. `bach21cache/`, `bach21data/` and
`bach21.log` are ignored by git.

## Settings

The `[note]` section of `config.ini` holds the settings the scripts use, each explained by a comment
there, and `DEFAULTS` in `config.py` lists every key with its default.
[docs/configuration.md](docs/configuration.md) describes every key.

- A changed model or training setting takes effect the next time you train with `--retrain`.
- A changed sampling setting (`temperature`, `rhythm_temperature`, `top_p`, `motif_threshold`) takes
  effect at the next generation, without retraining. `python evaluate.py bach violin --temperature 0.9`
  shows what a value does to the music before you change it.
- `hidden_size`, `tie_weights`, `add_causal_attn`, `attn_heads`, `number_of_steps`,
  `structured_embedding`, `pitch_relations` and `pattern_pointers` are part of the trained model, so
  change them only together with `--retrain`. With other values an existing model either fails to load
  or runs differently from how it was trained.
- `extra_composers` in `[note]` can add fitting parts of other composers to the training data (for
  example `handel, vivaldi`). It is empty by default: the loss gets better, the generated music does
  not (see [docs/results.md](docs/results.md)).
- The token files are only built once; delete `bach21data/<composer>/` to rebuild them after changing
  `data.py` or the cache.

To fine-tune a trained model with LoRA instead of training from scratch, set `lora_enable = true`,
`lora_peft_only = true` and `base_model = <path to a note_model.torch>` in `[note]`, then train with
`--retrain`. The result replaces `note_model.torch` in the target folder, so point `base_model` at a copy
if the base model lives in that same folder.

## Results

With the default settings, training on Bach's violin parts reaches a validation loss of about 2.71 nats
per token on held-out real violin parts. The previous defaults (the same network without the music-aware
layers) reach 2.85 on the same parts and take about six times as long to train. Most of the gain comes
from letting the network see and score intervals between notes.

The generated pieces are a more mixed story. Sampling melody and rhythm with separate temperatures makes
their rhythm as varied as real Bach's and more than halves the number of pieces that get stuck repeating
one figure. The music-aware network itself does not generate better pieces than the plain one: it repeats
single pitches less, but it loses the key and falls into loops more easily. With the default settings
about 1 piece in 12 gets stuck (try another `--seed`), the same as before, and 84% of the notes lie in the
best-fitting major scale, against 85% before and 91% for real Bach.

The full measurements, what each change contributes, and how to go back to the plain network are in
[docs/results.md](docs/results.md); `python evaluate.py bach violin --samples 144 --length 1000`
reproduces them for your own model.

## Reproducibility

Training is seeded (`TRAIN_SEED = 0` in `model.py`), so the part order, the train/validation split, the
initial weights and the training windows are the same on every run. Generation is seeded by `--seed`. The
cache files are sorted by name before they are shuffled, so the split is the same on Windows, Linux and
macOS. The losses can still differ in the last digits between CPU and GPU and between PyTorch versions.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| "Python 3.11 or newer was not found" | Install Python from python.org (tick "Add python.exe to PATH" or keep the `py` launcher) and run the script again. |
| Installing the dependencies fails | Check the internet connection and run again. Deleting `%USERPROFILE%\.venvs\bach21` starts from scratch. |
| "There is no trained model yet" | Run `train_bach_violin.bat` (or `train.py`) first. |
| "... already exists, skipping training" | A model is already there. Pass `--retrain` to train a new one. |
| Loading the model fails with a size mismatch | `config.ini` changed after training. Train again with `--retrain`, or restore the old settings. |
| Training is slow | Lower `number_of_epochs` or `hidden_size` in `[note]` (the epoch table in [docs/results.md](docs/results.md) shows what fewer epochs cost), or use a GPU build of PyTorch. |
| A piece is full of rests, or keeps repeating one figure | The model continues in the manner of its prompt, the 128 real notes it was primed with, and some prompts are sparse or repetitive accompaniment passages. Try another `--seed`; `--keep-prompt` writes the prompt too, so you can hear what it was given. |

## Known limitations

- The output is always a single violin line in treble clef and 4/4, whatever the model was trained on.
- Chords are not modelled: the cache keeps only single notes and rests.
- Durations are rounded to a 32nd-note grid, so tuplets are approximated.
- The model only sees the last 128 tokens, so there is no long-range form.
- Generated pieces hold a key less firmly than real Bach, and about 1 in 12 gets stuck repeating one
  figure (see [Results](#results)).
- All parts of one instrument in a score are stored as a single line, so two violins in one score are
  not kept apart.

## License

GPL-3.0, see [LICENSE](LICENSE).

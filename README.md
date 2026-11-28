# bach21

bach21 teaches a recurrent neural network to compose classical music. It learns from the scores in
[music21](https://web.mit.edu/music21/)'s corpus, for example every violin part Bach wrote, and then
writes new music in that style as MIDI and MusicXML.

## Quick start on Windows

You need [Python](https://www.python.org/downloads/) 3.11 or newer. Then, in the project folder:

1. Double-click **`train_bach_violin.bat`**, or run it from a terminal. It trains the model on Bach's
   violin parts. With the default settings this takes about half an hour on a 4-core CPU.
2. Double-click **`generate_bach_violin.bat`**. It writes a new 1000-note piece to
   `bach21data\bach\violin\midi_output.mid` and `xml_output.musicxml`.

The first time either script runs, it creates a virtual environment in your user folder, at
`%USERPROFILE%\.venvs\bach21` (for example `C:\Users\<you>\.venvs\bach21`), and installs everything from
`requirements.txt` into it. The project folder stays clean and nothing goes into your system Python. Later
runs reuse it and only reinstall when `requirements.txt` changes. To start over, delete that folder. To put
the environment somewhere else, set `BACH21_VENV_DIR` to the folder you want before running a script.

Both scripts pass extra options on to Python:

```bat
train_bach_violin.bat --retrain              rem train again although a model exists
generate_bach_violin.bat --seed 7            rem another piece (the same seed gives the same piece)
generate_bach_violin.bat --length 500        rem a shorter piece
generate_bach_violin.bat --keep-prompt       rem also write the real Bach notes the model was primed with
```

PyTorch from PyPI is the CPU build. To train on an NVIDIA GPU, set `BACH21_TORCH_INDEX_URL` (for example
to `https://download.pytorch.org/whl/cu128`) before the first run, or delete `%USERPROFILE%\.venvs\bach21` and run again with it
set.

## Running from Python directly (any OS)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip install -r requirements.txt
.venv/bin/python train.py bach violin
.venv/bin/python generate.py bach violin --length 1000
```

`train.py COMPOSER [INSTRUMENT ...]` and `generate.py COMPOSER [INSTRUMENT ...]` work for any composer and
instrument in the corpus cache (`python train.py bach cello`, `python train.py mozart violin`, ...). Without
an instrument they use every part of the composer. `python model.py bach violin` trains and then generates
in one go. Add `--help` to see every option.

## How it works

1. **Corpus cache** (`cache.py`). Every score is reduced to the pitches and durations of each instrument
   and stored as JSON in `bach21cache/`. A ready-made cache ships as `bach21cache.zip` and is unpacked
   automatically. `python cache.py --external C:\midi` rebuilds it with a folder of your own scores added
   (one sub-folder per composer).
2. **Tokens** (`data.py`). Each note becomes one token that combines pitch and duration, such as `C5_0.5`
   (C5, an eighth note) or `RST_1` (a quarter rest). Durations are rounded to 32nd notes, rests are merged
   and capped at a whole note. Each part becomes one line of `bach21data/<composer>/<instrument>/note_input.txt`.
3. **Training data** (`sequences.py`). The parts are split 80/20 into training and validation, by whole
   part. Training parts are also used transposed by up to 6 semitones, and the composer's other parts that
   fit the instrument's range are added (transposed by up to 4), as long as they do not double a violin
   part. Validation is always the real, untransposed violin parts. Training uses windows of 128 tokens.
4. **Model** (`network.py`). Embedding, one LSTM layer, a causal self-attention layer and an output layer
   that shares its weights with the embedding (about 600k weights with the default settings). It predicts
   the next token at every position.
5. **Training** (`model.py`). AdamW with weight decay and dropout. After every epoch the model with the
   lowest validation loss is kept; when an epoch does not improve, training goes back to the best weights
   with a quarter of the learning rate, and after 3 such epochs in a row it stops.
6. **Generation** (`model.py`, `generate.py`). The model is primed with 128 real notes from a random place
   in the corpus and then samples one token at a time at temperature 0.85. Only the new notes are written,
   unless `--keep-prompt` is given.

## Files

| File | What it does |
| --- | --- |
| `train_bach_violin.bat`, `generate_bach_violin.bat` | Windows entry points (see above) |
| `scripts/setup_venv.bat` | Creates the venv in `%USERPROFILE%\.venvs\bach21` and installs the dependencies; used by both `.bat` files |
| `train.py`, `generate.py` | Command-line training and generation |
| `config.ini` | Model, training and sampling settings, one section per model kind; `[note]` is the one used |
| `config.py` | Reads `config.ini` (every key and its default is listed in `DEFAULTS`), project paths, logging, seeding |
| `cache.py` | Builds or unpacks the corpus cache |
| `data.py` | Cache to token files, and generated tokens to MIDI / MusicXML |
| `sequences.py` | Train/validation split, transposition, extra parts, vocabulary, training windows |
| `network.py` | The PyTorch network |
| `model.py` | Training loop, sampling, and the `Worker` class that ties one model's files together |
| `motifs.py` | Frequent corpus motifs and the optional motif fallback while sampling (off for `[note]`) |
| `lora.py` | LoRA adapters for fine-tuning a trained model (off by default) |
| `entropy.py` | Entropy analysis of the corpus pitch sequences: `python entropy.py bach violin` |

Everything a run produces goes to `bach21data/<composer>/<instrument>/`:

| File | Contents |
| --- | --- |
| `note_input.txt`, `pitch_input.txt`, `duration_input.txt` | Training tokens, one part per line |
| `note_model.torch`, `note_vocabulary.json` | The trained model and its token list |
| `note_log.csv` | Training and validation loss per epoch (`best` = 1 where the model was saved) |
| `note_output.txt` | The generated tokens |
| `midi_output.mid`, `xml_output.musicxml` | The generated piece |

A log of every run is appended to `bach21.log` in the project folder.

## Settings

The `[note]` section of `config.ini` holds the settings the scripts use, each explained by a comment
there, and `DEFAULTS` in `config.py` lists every key with its default. A changed setting takes effect the
next time you train with `--retrain`. The token files are only built once; delete
`bach21data/<composer>/` to rebuild them after changing `data.py`.

To fine-tune a trained model with LoRA instead of training from scratch, set `lora_enable = true`,
`lora_peft_only = true` and `base_model = <path to a note_model.torch>` in `[note]`.

## Results

With the default settings, training on Bach's violin parts reaches a validation loss of about 3.15
(nats per token, on held-out real violin parts).

## License

GPL-3.0, see [LICENSE](LICENSE).

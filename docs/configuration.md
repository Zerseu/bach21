# Configuration

Settings live in [`config.ini`](../config.ini), one section per model kind. `train.py` and `generate.py`
use `[note]` unless `--kind` names another section. A key a section leaves out takes its default from
`DEFAULTS` in [`config.py`](../config.py); the type of that default decides how the value is read
(`true`/`false`, whole number, decimal number or text). Keys marked *required* have no default. Unknown
keys are ignored.

When a setting takes effect:

- **Model** settings are part of the trained model. Change them only together with `--retrain`: an
  existing model either fails to load with other values (`hidden_size`, `tie_weights`,
  `add_causal_attn`) or loads but runs differently from how it was trained (`attn_heads`,
  `number_of_steps`).
- **Training** settings are used when a model is trained (`--retrain` for an existing one).
- **Sampling** settings are read at every generation and need no retraining.

## Network

| Key | `[note]` | Default | When | Meaning |
| --- | --- | --- | --- | --- |
| `number_of_steps` | 128 | required | model | Tokens per training window, and the context the model sees while generating. Also the prompt length. |
| `hidden_size` | 192 | required | model | Width of the embedding, the LSTM and the attention layer. The main lever for model size: 256 gives about 1M weights, 192 about 600k. |
| `tie_weights` | true | false | model | The output layer reuses the embedding matrix. Saves vocabulary × hidden_size weights and helps rare tokens. |
| `add_causal_attn` | (true) | true | model | Adds the causal self-attention layer after the LSTM. |
| `attn_heads` | (4) | 4 | model | Number of attention heads; must divide `hidden_size`. |
| `dropout` | 0.4 | 0.25 | training | Dropout after the embedding. |
| `output_dropout` | 0.5 | 0.0 | training | Dropout before the output layer. |

## Training

| Key | `[note]` | Default | Meaning |
| --- | --- | --- | --- |
| `batch_size` | 64 | required | Windows per optimiser step. |
| `number_of_epochs` | 50 | required | Maximum number of passes over the training windows. |
| `window_stride` | 32 | 0 | Distance between the starts of consecutive windows. 0 means `number_of_steps // 4`. A smaller stride gives more, more overlapping windows. |
| `early_stopping_patience` | 3 | 0 | Stop after this many epochs in a row without a lower validation loss. 0 always runs every epoch. |
| `learning_rate` | (0.001) | 0.001 | AdamW learning rate. |
| `weight_decay` | 0.2 | 0.0 | AdamW weight decay. |
| `lr_decay` | 0.25 | 0.0 | After an epoch without improvement, reload the best weights and multiply the learning rate by this. Only values between 0 and 1 (exclusive) do anything. |
| `grad_clip` | (0) | 0.0 | Clip the gradient norm to this value; 0 turns it off. |

## Training data

| Key | `[note]` | Default | Meaning |
| --- | --- | --- | --- |
| `transpose` | 6 | 0 | Also train on copies of each training part shifted by up to ± this many semitones, as long as they stay in the instrument's range. |
| `extra_training_data` | true | false | Add the composer's other parts that fit the instrument's range and do not double one of its parts. Only when an instrument is given. |
| `transpose_extra` | 4 | 0 | Transposition range for those extra parts. |

These three only apply to the `note` and `pitch` kinds. Validation always uses the real, untransposed
parts. More data makes each epoch slower: with the `[note]` values an epoch on 4 CPU cores takes about
3 minutes.

## Sampling

| Key | `[note]` | Default | Meaning |
| --- | --- | --- | --- |
| `temperature` | 0.85 | required | Divides the logits before sampling. Below 1 the model sticks closer to its most likely choices, above 1 it takes more risks. 0.85 keeps the key as well as real Bach does; 1.0 drifts more. |
| `top_p` | 1.0 | 1.0 | Nucleus sampling: draw only from the smallest set of tokens whose probabilities add up to `top_p`. 1 turns it off. |
| `motif_threshold` | 0 | 0.10 | When the most likely token has at most this probability, continue a frequent corpus motif instead. 0 turns it off. |

## LoRA fine-tuning

| Key | `[note]` | Default | Meaning |
| --- | --- | --- | --- |
| `lora_enable` | false | false | Wrap weights with low-rank adapters. A model trained this way must be loaded with it on too. |
| `lora_peft_only` | false | false | Train only the adapters and biases; the base weights stay frozen. |
| `base_model` | (empty) | (empty) | Path of a `note_model.torch` to start from. Needed for LoRA; without LoRA it continues training a model from those weights. Its shape settings and vocabulary must match. |
| `lora_r` | (8) | 8 | Rank of the adapters. |
| `lora_alpha` | (16) | 16 | The adapter update is scaled by `lora_alpha / lora_r`. |
| `lora_dropout` | (0.05) | 0.05 | Accepted for compatibility; the adapters do not apply it. |
| `lora_targets` | (`ih,hh,out,emb`) | `ih,hh,out,emb` | Which weights get adapters: LSTM input (`ih`) and recurrent (`hh`) weights, the output layer (`out`), and an extra identity projection after the embedding (`emb`). |

Values in parentheses are not set in `[note]` and come from the default.

## Constants in the code

A few settings are constants rather than config keys:

| Constant | Value | Where | Meaning |
| --- | --- | --- | --- |
| `TRAIN_SEED` | 0 | `model.py` | Seed for the part order, split, initial weights and batch order. |
| `TRAIN_RATIO` | 0.8 | `model.py` | Share of the tokens, in whole parts, used for training. |
| `DEFAULT_LENGTH` | 1000 | `model.py` | Default `--length`. |
| `DURATION_GRID` | 0.125 | `data.py` | Durations are rounded to this many quarter notes (a 32nd note). |
| `MAX_REST_LENGTH` | 4.0 | `data.py` | Longer rests are shortened to a whole note. |
| `FILTER_PARTS` | False | `data.py` | Drop rest-heavy and near-duplicate parts (slow). |
| `FILTER_RESTS` | True | `data.py` | Leave rests out of `pitch_input.txt` / `duration_input.txt`. |
| `OVERLAP_RUN` | 12 | `sequences.py` | An extra part sharing this many consecutive tokens with an instrument part is left out. |
| `MOTIF_MIN_LENGTH`, `MOTIF_MAX_LENGTH` | 4, 8 | `motifs.py` | Lengths of the motifs the fallback uses. |

Changing anything in `data.py` needs the token files rebuilt: delete `bach21data/<composer>/`.

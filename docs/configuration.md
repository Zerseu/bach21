# Configuration

Settings live in [`config.ini`](../config.ini), one section per model kind. `train.py`, `generate.py` and
`evaluate.py` use `[note]` unless `--kind` names another section. A key a section leaves out takes its default from
`DEFAULTS` in [`config.py`](../config.py); the type of that default decides how the value is read
(`true`/`false`, whole number, decimal number or text). Keys marked *required* have no default. Unknown
keys are ignored.

When a setting takes effect:

- **Model** settings are part of the trained model. Change them only together with `--retrain`: an
  existing model either fails to load with other values (`hidden_size`, `tie_weights`,
  `add_causal_attn`, `structured_embedding`, `pitch_relations`, `pattern_pointers`) or loads but runs
  differently from how it was trained (`attn_heads`, `number_of_steps`).
- **Training** settings are used when a model is trained (`--retrain` for an existing one).
- **Sampling** settings are read at every generation and need no retraining.

## Network

| Key | `[note]` | Default | When | Meaning |
| --- | --- | --- | --- | --- |
| `number_of_steps` | 128 | required | model | Tokens per training window, and the context the model sees while generating. Also the prompt length. |
| `hidden_size` | 192 | required | model | Width of the embedding, the LSTM and the attention layer. The main lever for model size: 256 gives about 1M weights, 192 about 640k. |
| `tie_weights` | true | false | model | The output layer reuses the embedding matrix. Saves vocabulary × hidden_size weights and helps rare tokens. |
| `add_causal_attn` | (true) | true | model | Adds the causal self-attention layer after the LSTM. |
| `attn_heads` | (4) | 4 | model | Number of attention heads; must divide `hidden_size`. |
| `structured_embedding` | true | false | model | A token's vector is the sum of a part of its own, a pitch part shared by all tokens of that pitch and a duration part shared by all tokens of that duration. Rare tokens then learn from common ones. Note model only. |
| `pitch_relations` | true | false | model | Adds a pitch-class part to the token vector, the interval from the previous note as an input, and output scores per interval from the last note. What the network learns about melodic motion then holds in every key. Note model only. |
| `pattern_pointers` | false | false | model | Experimental. Adds output scores for repeating the pitch, the interval or the duration found 1 to 12 notes back. It lowers the held-out loss by about 0.03 and costs about 14% more time per batch, but the generated music was not better (see [results.md](results.md#options-that-lower-the-loss-further)), so it is off. Note model only. |
| `dropout` | 0.4 | 0.25 | training | Dropout after the embedding. |
| `output_dropout` | 0.5 | 0.0 | training | Dropout before the output layer. |

## Training

| Key | `[note]` | Default | Meaning |
| --- | --- | --- | --- |
| `batch_size` | 64 | required | Windows per optimiser step. |
| `number_of_epochs` | 6 | required | Number of epochs. With `lr_schedule = cosine` the run always has this length; otherwise it is the maximum. |
| `random_windows` | true | false | Train on windows that start at random places, redrawn for every batch. One epoch is then about one visit per training token (`tokens / (number_of_steps * batch_size)` batches). When false, the windows are fixed, one every `window_stride` tokens, and an epoch visits them all. |
| `window_stride` | 32 | 0 | Distance between the starts of consecutive fixed windows. 0 means `number_of_steps // 4`. The validation windows always use it; the training windows only when `random_windows` is false. |
| `early_stopping_patience` | 0 | 0 | Stop after this many epochs in a row without a lower validation loss. 0 always runs every epoch. |
| `learning_rate` | 0.002 | 0.001 | AdamW learning rate (the peak, with the cosine schedule). |
| `lr_schedule` | cosine | constant | `constant` keeps the learning rate fixed (apart from `lr_decay`). `cosine` raises it from 0 to `learning_rate` over `lr_warmup_steps` batches and then lowers it to 0 along a cosine curve by the end of the last epoch. |
| `lr_warmup_steps` | 100 | 0 | Batches of linear warm-up at the start of the cosine schedule. |
| `weight_decay` | 0.2 | 0.0 | AdamW weight decay. |
| `lr_decay` | (0) | 0.0 | With `lr_schedule = constant`: after an epoch without improvement, reload the best weights and multiply the learning rate by this. Only values between 0 and 1 (exclusive) do anything. Ignored with the cosine schedule. |
| `grad_clip` | 1.0 | 0.0 | Clip the gradient norm to this value; 0 turns it off. |

Whatever the schedule, the model saved at the end is the one from the epoch with the lowest validation loss.

## Training data

| Key | `[note]` | Default | Meaning |
| --- | --- | --- | --- |
| `transpose` | 6 | 0 | Also train on copies of each training part shifted by up to ± this many semitones, as long as they stay in the instrument's range. |
| `extra_training_data` | true | false | Add other parts that fit the instrument's range, use only tokens the instrument's data already has, and do not double one of its parts: the composer's other instruments (when an instrument is given) and the composers in `extra_composers`. |
| `extra_composers` | (empty) | (empty) | Other composers in the corpus cache whose fitting parts are added as well, comma-separated, for example `handel, vivaldi`. Needs `extra_training_data = true`. With Handel and Vivaldi the held-out loss falls by another 0.15 and training takes 75% longer, but the generated pieces were not better (see [results.md](results.md#options-that-lower-the-loss-further)), so the default trains on the model's own composer only. |
| `transpose_extra` | 4 | 0 | Transposition range for all those extra parts. |

These four only apply to the `note` and `pitch` kinds. Validation always uses the real, untransposed
parts of the model's own composer and instrument, so the validation loss always measures how well the
model predicts that music, whatever else it trained on.

## Sampling

| Key | `[note]` | Default | Meaning |
| --- | --- | --- | --- |
| `temperature` | 0.8 | required | Below 1 the model sticks closer to its most likely choices, above 1 it takes more risks. With `rhythm_temperature = 0` it applies to whole tokens (the logits are divided by it); otherwise only to the choice of pitch. Lower values keep the melody in key better; too low and it repeats itself. |
| `rhythm_temperature` | 1.0 | 0.0 | Note model only. Above 0, a token is sampled in three steps: rest or note, then the pitch, then the duration. The first and the last use this temperature, the pitch uses `temperature`. Rhythm needs a higher temperature than melody: at one shared temperature that keeps the melody in key, the generated durations are much less varied than real ones. 0 samples whole tokens with `temperature` alone. |
| `top_p` | 1.0 | 1.0 | Nucleus sampling: draw only from the smallest set of choices whose probabilities add up to `top_p` (tokens, or pitches when `rhythm_temperature` is on). 1 turns it off. |
| `motif_threshold` | 0 | 0.10 | When the most likely token has at most this probability, continue a frequent corpus motif instead. 0 turns it off. |

## LoRA fine-tuning

| Key | `[note]` | Default | Meaning |
| --- | --- | --- | --- |
| `lora_enable` | false | false | Wrap weights with low-rank adapters. A model trained this way must be loaded with it on too. |
| `lora_peft_only` | false | false | Train only the adapters and biases; the base weights stay frozen. |
| `base_model` | (empty) | (empty) | Path of a `note_model.torch` to start from. Needed for LoRA; without LoRA it continues training a model from those weights. Its shape settings and vocabulary must match. |
| `lora_r` | (8) | 8 | Rank of the adapters. |
| `lora_alpha` | (16) | 16 | The adapter update is scaled by `lora_alpha / lora_r`. |
| `lora_dropout` | (0.05) | 0.05 | Dropout on the adapter matrix A during training (the adapters see the weight, not the layer input, so it drops entries of the low-rank update). |
| `lora_targets` | (`ih,hh,out,emb`) | `ih,hh,out,emb` | Which weights get adapters: LSTM input (`ih`) and recurrent (`hh`) weights, the output layer (`out`), and an extra identity projection after the embedding (`emb`). |

Values in parentheses are not set in `[note]` and come from the default.

## Constants in the code

A few settings are constants rather than config keys:

| Constant | Value | Where | Meaning |
| --- | --- | --- | --- |
| `TRAIN_SEED` | 0 | `model.py` | Seed for the part order, split, initial weights and training windows. |
| `TRAIN_RATIO` | 0.8 | `model.py` | Share of the tokens, in whole parts, used for training. |
| `DEFAULT_LENGTH` | 1000 | `model.py` | Default `--length`. |
| `DURATION_GRID` | 0.125 | `data.py` | Durations are rounded to this many quarter notes (a 32nd note). |
| `MAX_REST_LENGTH` | 4.0 | `data.py` | Longer rests are shortened to a whole note. |
| `FILTER_PARTS` | False | `data.py` | Drop rest-heavy and near-duplicate parts (slow). |
| `FILTER_RESTS` | True | `data.py` | Leave rests out of `pitch_input.txt` / `duration_input.txt`. |
| `OVERLAP_RUN` | 12 | `sequences.py` | An extra part sharing this many consecutive tokens with an instrument part is left out. |
| `MAX_INTERVAL` | 24 | `network.py` | `pitch_relations` clips intervals to this many semitones up or down. |
| `POINTER_SPAN` | 12 | `network.py` | How many notes back `pattern_pointers` can point. |
| `MOTIF_MIN_LENGTH`, `MOTIF_MAX_LENGTH` | 4, 8 | `motifs.py` | Lengths of the motifs the fallback uses. |

Changing anything in `data.py` needs the token files rebuilt: delete `bach21data/<composer>/`.

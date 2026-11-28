"""Project paths, the per-model settings read from config.ini, logging and seeding.

config.ini has one section per model kind (``note``, ``pitch``, ``duration``). Every key a section
leaves out falls back to the default in ``DEFAULTS`` below, so a section only needs the values that
differ. Only the ``note`` model is used by the training and generation scripts; the ``pitch`` and
``duration`` sections are kept for experiments with separate pitch and rhythm models.
"""
import configparser
import os
import random
from sys import stdout

import numpy as np
import torch

# All paths are relative to the project folder, so the scripts work from any current directory.
ROOT: str = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH: str = os.path.join(ROOT, 'config.ini')
LOG_PATH: str = os.path.join(ROOT, 'bach21.log')
CACHE_ROOT: str = os.path.join(ROOT, 'bach21cache')
CACHE_ZIP: str = os.path.join(ROOT, 'bach21cache.zip')
DATA_ROOT: str = os.path.join(ROOT, 'bach21data')

# Key -> default value. The type of the default decides how the value in config.ini is parsed.
# None marks a required key.
DEFAULTS: dict = {
    # Network and training
    'number_of_steps': None,          # tokens per training window and generation context
    'batch_size': None,
    'hidden_size': None,              # embedding, LSTM and attention width
    'number_of_epochs': None,
    'window_stride': 0,               # 0 means number_of_steps // 4
    'random_windows': False,          # train on windows at random starts instead of one every window_stride
    'early_stopping_patience': 0,     # stop after this many epochs without improvement; 0 never stops early
    'learning_rate': 1e-3,
    'weight_decay': 0.0,              # AdamW weight decay
    'grad_clip': 0.0,                 # gradient norm clipping; 0 turns it off
    'lr_decay': 0.0,                  # on a non-improving epoch reload the best weights and scale the LR by this
    'lr_schedule': 'constant',        # 'constant', or 'cosine': warm up, then fall to 0 over number_of_epochs
    'lr_warmup_steps': 0,             # batches of linear warm-up at the start of the cosine schedule
    'dropout': 0.25,                  # after the embedding
    'output_dropout': 0.0,            # before the output layer
    'tie_weights': False,             # share the input embedding and output layer weights
    'add_causal_attn': True,          # causal self-attention layer after the LSTM
    'attn_heads': 4,
    'structured_embedding': False,    # token vector = token part + shared pitch part + shared duration part
    'pitch_relations': False,         # pitch-class embedding, interval input and interval output scores
    'pattern_pointers': False,        # scores for repeating the pitch / interval / duration of k notes ago
    # LoRA fine-tuning
    'lora_enable': False,
    'lora_r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.05,
    'lora_targets': 'ih,hh,out,emb',
    'lora_peft_only': False,          # train only the adapters, biases and LayerNorm
    'base_model': '',                 # model.torch to start from (needed for LoRA fine-tuning)
    # Training data
    'transpose': 0,                   # also train on copies shifted by up to +/- this many semitones
    'extra_training_data': False,     # add the composer's other parts that fit the instrument's range
    'extra_composers': '',            # other composers whose fitting parts are added too, comma-separated
    'transpose_extra': 0,             # transposition for those extra parts
    # Sampling
    'temperature': None,
    'rhythm_temperature': 0.0,        # note model: separate temperature for rests and durations; 0 = off
    'top_p': 1.0,                     # nucleus sampling; 1 turns it off
    'motif_threshold': 0.10,          # continue a corpus motif when the top probability is at or below this
}


class Config:
    """Settings from config.ini, as ``Config().config[kind][key]``."""

    def __init__(self, config: str = CONFIG_PATH):
        parser = configparser.ConfigParser()
        if not parser.read(config):
            raise FileNotFoundError(f'Config file not found: {config}')
        self.config: dict[str, dict] = {}
        for section in parser.sections():
            self.config[section] = {key: Config._read(parser, section, key, default)
                                    for key, default in DEFAULTS.items()}

    def __getitem__(self, kind: str) -> dict:
        """All settings of one section, for example ``Config()['note']['hidden_size']``."""
        return self.config[kind]

    @staticmethod
    def _read(parser: configparser.ConfigParser, section: str, key: str, default):
        """One value of a section, parsed with the type of its default (bool, int, float or str).

        Raises KeyError when a required key (default None) is missing. Keys in config.ini that are not
        in DEFAULTS are ignored.
        """
        if default is None:
            if not parser.has_option(section, key):
                raise KeyError(f'config.ini [{section}] is missing the required key "{key}"')
            # Required keys: temperature is a float, the rest are integers.
            return parser.getfloat(section, key) if key == 'temperature' else parser.getint(section, key)
        if isinstance(default, bool):
            return parser.getboolean(section, key, fallback=default)
        if isinstance(default, int):
            return parser.getint(section, key, fallback=default)
        if isinstance(default, float):
            return parser.getfloat(section, key, fallback=default)
        return parser.get(section, key, fallback=default).strip()


def log(*values: object):
    """Prints to the console and appends the same line to bach21.log (the file only ever grows)."""
    print(*values, file=stdout, flush=True)
    with open(LOG_PATH, 'at', encoding='utf-8') as file:
        print(*values, file=file, flush=True)


def seed_everything(seed: int):
    """Seeds Python, NumPy and PyTorch so data order, initial weights and sampling are reproducible.

    Results are reproducible on the same machine and library versions; CPU and GPU runs, or different
    PyTorch versions, can still differ in the last digits and so in later sampled tokens.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

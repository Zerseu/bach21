import configparser
from sys import stdout


class Config:
    def __init__(self, config: str = 'config.ini'):
        parser = configparser.ConfigParser()
        parser.read(config)
        self.config = {}
        sections = parser.sections()
        for s in range(len(sections)):
            dictionary = dict(number_of_steps=parser.getint(sections[s], 'number_of_steps'),
                              batch_size=parser.getint(sections[s], 'batch_size'),
                              hidden_size=parser.getint(sections[s], 'hidden_size'),
                              number_of_epochs=parser.getint(sections[s], 'number_of_epochs'),
                              temperature=parser.getfloat(sections[s], 'temperature'),
                              lora_enable=parser.getboolean(sections[s], 'lora_enable', fallback=False),
                              lora_r=parser.getint(sections[s], 'lora_r', fallback=8),
                              lora_alpha=parser.getint(sections[s], 'lora_alpha', fallback=16),
                              lora_dropout=parser.getfloat(sections[s], 'lora_dropout', fallback=0.05),
                              lora_targets=parser.get(sections[s], 'lora_targets', fallback='ih,hh,out,emb'),
                              add_causal_attn=parser.getboolean(sections[s], 'add_causal_attn', fallback=True),
                              attn_heads=parser.getint(sections[s], 'attn_heads', fallback=4),
                              lora_peft_only=parser.getboolean(sections[s], 'lora_peft_only', fallback=False),
                              base_model=parser.get(sections[s], 'base_model', fallback=''),
                              window_stride=parser.getint(sections[s], 'window_stride', fallback=0),
                              early_stopping_patience=parser.getint(sections[s], 'early_stopping_patience', fallback=0),
                              transpose=parser.getint(sections[s], 'transpose', fallback=0),
                              tie_weights=parser.getboolean(sections[s], 'tie_weights', fallback=False),
                              dropout=parser.getfloat(sections[s], 'dropout', fallback=0.25),
                              output_dropout=parser.getfloat(sections[s], 'output_dropout', fallback=0.0),
                              learning_rate=parser.getfloat(sections[s], 'learning_rate', fallback=1e-3),
                              weight_decay=parser.getfloat(sections[s], 'weight_decay', fallback=0.0),
                              grad_clip=parser.getfloat(sections[s], 'grad_clip', fallback=0.0),
                              top_p=parser.getfloat(sections[s], 'top_p', fallback=1.0),
                              extra_training_data=parser.getboolean(sections[s], 'extra_training_data', fallback=False),
                              transpose_extra=parser.getint(sections[s], 'transpose_extra', fallback=0),
                              lr_decay=parser.getfloat(sections[s], 'lr_decay', fallback=0.0),
                              motif_threshold=parser.getfloat(sections[s], 'motif_threshold', fallback=0.10))
            self.config[sections[s]] = dictionary

    def __str__(self):
        result = ''
        result += f"ns_{self.config['pitch']['number_of_steps']}_"
        result += f"bs_{self.config['pitch']['batch_size']}_"
        result += f"hs_{self.config['pitch']['hidden_size']}_"
        result += f"ne_{self.config['pitch']['number_of_epochs']}_"
        result += f"tmp_{self.config['pitch']['temperature']:.1f}"
        result += f"_lora{int(self.config['pitch'].get('lora_enable', False))}r{self.config['pitch'].get('lora_r', 8)}"
        return result


def log(*values: object):
    print(*values, file=stdout, flush=True)
    with open('bach21.log', 'at') as file:
        print(*values, file=file, flush=True)

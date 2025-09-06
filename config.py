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
                              lora_enable=parser.getboolean(sections[s], 'lora_enable', fallback=True),
                              lora_r=parser.getint(sections[s], 'lora_r', fallback=8),
                              lora_alpha=parser.getint(sections[s], 'lora_alpha', fallback=16),
                              lora_dropout=parser.getfloat(sections[s], 'lora_dropout', fallback=0.05),
                              lora_targets=parser.get(sections[s], 'lora_targets', fallback='ih,hh,out,emb'),
                              add_causal_attn=parser.getboolean(sections[s], 'add_causal_attn', fallback=True),
                              attn_heads=parser.getint(sections[s], 'attn_heads', fallback=4),
                              lora_peft_only=parser.getboolean(sections[s], 'lora_peft_only', fallback=True))
            self.config[sections[s]] = dictionary

    def __str__(self):
        result = ''
        result += f"ns_{self.config['pitch']['number_of_steps']}_"
        result += f"bs_{self.config['pitch']['batch_size']}_"
        result += f"hs_{self.config['pitch']['hidden_size']}_"
        result += f"ne_{self.config['pitch']['number_of_epochs']}_"
        result += f"tmp_{self.config['pitch']['temperature']:.1f}"
        result += f"_lora{int(self.config['pitch'].get('lora_enable', True))}r{self.config['pitch'].get('lora_r', 8)}"
        return result


def log(*values: object):
    print(*values, file=stdout, flush=True)
    with open('bach21.log', 'at') as file:
        print(*values, file=file, flush=True)

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
                              temperature=parser.getfloat(sections[s], 'temperature'))
            self.config[sections[s]] = dictionary

    def __str__(self):
        result = ''
        result += f"ns_{self.config['pitch']['number_of_steps']}_"
        result += f"bs_{self.config['pitch']['batch_size']}_"
        result += f"hs_{self.config['pitch']['hidden_size']}_"
        result += f"ne_{self.config['pitch']['number_of_epochs']}_"
        result += f"tmp_{self.config['pitch']['temperature']:.1f}"
        return result


def log(*values: object):
    print(*values, file=stdout, flush=True)
    with open('bach21.log', 'at') as file:
        print(*values, file=file, flush=True)

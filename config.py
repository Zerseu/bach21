import configparser


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
                              motif_augmentation=parser.getboolean(sections[s], 'motif_augmentation'))
            self.config[sections[s]] = dictionary


def fprintf(*values: object):
    print(*values)
    with open('bach21.log', 'at') as log:
        print(*values, file=log, flush=True)

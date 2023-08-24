import configparser


class Config:
    def __init__(self, config: str = 'config.ini'):
        parser = configparser.ConfigParser()
        parser.read(config)
        self.config = []
        sections = parser.sections()
        for s in range(len(sections)):
            dictionary = dict(number_of_steps=parser.getint(sections[s], 'number_of_steps'),
                              batch_size=parser.getint(sections[s], 'batch_size'),
                              hidden_size=parser.getint(sections[s], 'hidden_size'),
                              number_of_epochs=parser.getint(sections[s], 'number_of_epochs'),
                              temperature=parser.getfloat(sections[s], 'temperature'))
            self.config.append(dictionary)

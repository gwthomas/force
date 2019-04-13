from argparse import ArgumentParser


class Configuration:
    def __init__(self, entries):
        ''' entries should be a list of triples, each consisting of
                (name,
                 possibilities,
                 default)
            possilibities should be a type or a list of values
            If default is None, the entry's value must be passed via the command line
        '''
        self.entries = entries
        self.parser = ArgumentParser()
        for name, possibilities, default in self.entries:
            if isinstance(possibilities, type):
                type_arg = possibilities
            elif isinstance(possibilities, list):
                type_arg = str
            else:
                raise RuntimeError('Invalid possibilities: {}'.format(possibilities))

            name_arg = name if default is None else '--' + name
            self.parser.add_argument(name_arg, type=type_arg, default=default)

    def parse(self):
        args = self.parser.parse_args()
        for name, possibilities, default in self.entries:
            val = getattr(args, name)
            if isinstance(possibilities, type):
                if not isinstance(val, possibilities):
                    raise TypeError('Invalid argument: {}. Expected type {}'.format(val, possibilities))
            elif isinstance(possibilities, list):
                if val not in possibilities:
                    raise ValueError('Invalid argument: {}. Options: {}'.format(val, possibilities))
        return args

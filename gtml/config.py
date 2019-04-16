from argparse import ArgumentParser


REQUIRED = '__REQUIRED__'

class Configuration:
    def __init__(self, entries):
        ''' entries should be a list of triples, each consisting of
                (name,
                 possibilities,
                 default)
            possibilities should be an x or a list/tuple of x, where x is a type or a specific value
            If default is REQUIRED, the entry's value must be passed via the command line
        '''
        self.entries = entries
        self.parser = ArgumentParser()
        for name, possibilities, default in self.entries:
            if isinstance(possibilities, type):
                type_arg = possibilities
            else:
                type_arg = None
                
            if default is REQUIRED:
                name_arg = name
                default_arg = None
            else:
                name_arg = '--' + name
                default_arg = default
            self.parser.add_argument(name_arg, type=type_arg, default=default_arg)

    def parse(self):
        args = self.parser.parse_args()
        for name, possibilities, default in self.entries:
            value = getattr(args, name)
            if type(possibilities) not in (list, tuple):
                possibilities = [possibilities]
            
            match = False
            for possibility in possibilities:
                if isinstance(possibility, type):
                    if isinstance(value, possibility):
                        match = True
                        break
                    
                    # Not already the right type, but try casting it
                    try:
                        value = possibility(value)
                        match = True
                        break
                    except:
                        pass
                else:
                    if value == possibility:
                        match = True
                        break
            if not match:
                raise ValueError('Invalid value: {}. Possibilities:', possibilities)
        return args

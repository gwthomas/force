from argparse import ArgumentParser


# To be used for boolean objects, because argparse does not play nicely
# with the built-in bool type
class boolean:
    def __init__(self, s):
        if s in (True, 1, 'True', '1', 'true', 'yes'):
            self.value = True
        elif s in (False, 0, 'False', '0', 'false', 'no'):
            self.value = False
        else:
            raise ValueError('Cannot convert {} to boolean'.format(s))

    def __bool__(self):
        return self.value

    def __repr__(self):
        return '<boolean value={}>'.format(self.value)


class ConfigItem:
    def __init__(self, name, accept):
        self.name = name
        self.accept = accept
        
    def _parse_name(self):
        raise NotImplementedError
        
    def _parse_type(self):
        if isinstance(self.accept, type):
            return self.accept
        else:
            return str
        
    def _parse_default(self):
        return None
        
    def add_parse_arg(self, parser):
        parser.add_argument(self._parse_name(),
                            type=self._parse_type(),
                            default=self._parse_default())
        
    def process(self, args):
        value = getattr(args, self.name)
        if isinstance(self.accept, type):
            if not isinstance(value, self.accept):
                try:
                    value = self.accept(value)
                except:
                    raise ValueError('Expected type {} for config item {}, but got value {}'.format(self.accept, self.name, value))
            setattr(args, self.name, value)
        elif type(self.accept) in (list, tuple):
            if value not in self.accept:
                raise ValueError('Invalid argument value {} for config item {}. Accepted options: {}'.format(value, self.name, self.accept))
            
        
class RequiredItem(ConfigItem):
    def _parse_name(self):
        return self.name


class OptionalItem(ConfigItem):
    def _parse_name(self):
        return '--' + self.name
    
    def process(self, args):
        if getattr(args, self.name) is not None:
            ConfigItem.process(self, args)


class DefaultingItem(OptionalItem):
    def __init__(self, name, accept, default):
        ConfigItem.__init__(self, name, accept)
        self.default = default
        
    def _parse_default(self):
        return self.default


class Config:
    def __init__(self, items):
        self.items = items
        
    def parse(self):
        parser = ArgumentParser()
        for item in self.items:
            item.add_parse_arg(parser)
        args = parser.parse_args()
        for item in self.items:
            item.process(args)
        return args
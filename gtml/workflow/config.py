import json
from pathlib import Path


class SpecialValue:
    pass
REQUIRED = SpecialValue()
OPTIONAL = SpecialValue()


class ConfigItem:
    # If default_or_special is not one of the special values REQUIRED, OPTIONAL,
    # then it will be treated as a default value
    def __init__(self, name, accepts, default_or_special):
        self.name = name
        self.accepts = accepts
        self.default_or_special = default_or_special

    def process(self, d):
        if self.name in d:
            value = d[self.name]
            if isinstance(self.accepts, type):
                if not isinstance(value, self.accepts):
                    try:
                        value = self.accepts(value)
                    except:
                        raise ValueError('Expected type {} for config item {}, but got value {}'.format(self.accepts, self.name, value))
            elif type(self.accepts) in (list, tuple):
                if value not in self.accepts:
                    raise ValueError('Invalid argument value {} for config item {}. Accepted options: {}'.format(value, self.name, self.accepts))
            return value
        else: # No value was given for this item
            if self.default_or_special is REQUIRED:
                raise ValueError('No value given for required config item {}'.format(self.name))
            elif self.default_or_special is OPTIONAL:
                return None
            else:
                return self.default_or_special

class Config:
    def __init__(self, items):
        self.items = items

    def parse_dict(self, d):
        return {item.name: item.process(d) for item in self.items}

    def parse_json_file(self, path):
        text = Path(path).read_text()
        return self.parse_dict(json.loads(text))

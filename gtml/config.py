import copy

from gtml.core import Enumeration


class Require:
    def __init__(self, template):
        assert isinstance(template, Enumeration) or isinstance(template, type)
        self.template = template

    def resolve(self, x):
        if isinstance(self.template, Enumeration):
            return self.template.resolve(x)
        elif isinstance(self.template, type):
            if not isinstance(x, self.template):
                raise ValueError('Invalid type: {}. Expected: {}'.format(type(x), self.template))
            return x
        else:
            raise ValueError('Invalid argument: {}'.format(x))


class Config:
    def __init__(self, description):
        self.d = copy.deepcopy(description)
        self.verified = False

    def __getitem__(self, key):
        return self.d[key]

    def update(self, d, verify=True, raise_on_extra=False):
        for key, value in d.items():
            if key in self.d:
                existing = self.d[key]
                if isinstance(existing, Require):
                    self.d[key] = existing.resolve(value)
                else:
                    self.d[key] = value
            else:
                print('Extra key:', format(key))
                if raise_on_extra:
                    raise RuntimeError('raise_on_extra triggered')

        if verify:
            self.verify()

    def verify(self):
        for key, value in self.d.items():
            if isinstance(value, Require):
                raise RuntimeError('Required item {} not specified'.format(key))
        self.verified = True

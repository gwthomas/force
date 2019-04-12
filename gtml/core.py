class Serializable:
    # Should return a list of strings, which are the names of the attributes
    # to be serialized (override this method in subclass)
    def _state_attrs(self):
        return []

    def state_dict(self):
        return {attr: getattr(self, attr) for attr in self._state_attrs()}

    def load_state_dict(self, d):
        for attr in self._state_attrs():
            setattr(self, attr, d[attr])


class Enumeration:
    def __init__(self, options):
        if isinstance(options, list) or isinstance(options, tuple):
            self.options = {x:x for x in options}
        elif isinstance(options, dict):
            self.options = options.copy()
        else:
            raise ValueError

    def resolve(self, x):
        if x in self.options:
            return self.options[x]
        else:
            raise ValueError('Invalid argument: {}. Options: {}'.format(x, tuple(self.options.keys())))

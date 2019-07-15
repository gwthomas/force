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


class StateContainer(Serializable):
    def __init__(self, value):
        self.value = value

    def _state_attrs(self):
        return ['value']

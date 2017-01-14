import numpy as np

class Data(object):
    def __init__(self, experiment, name, value=None):
        self._experiment = experiment
        self._name = name
        self._array = np.array(value)
        self.listeners = set()

    def get(self):
        return np.copy(self._array)

    def set(self, new_array):
        if isinstance(new_array, np.ndarray):
            self._array = new_array
        else:
            raise Exception('Assigning invalid type {} to data named {}'.format(type(new_array), self._name))

        self.fire()

    def apply(self, f):
        self.set(f(self._array))

    def save(self, file):
        with open(file, 'wb') as f:
            np.save(f, self._array)

    def load(self, file):
        with open(file, 'rb') as f:
            self.set(np.load(f))

    def fire(self):
        for listener in self.listeners:
            listener.callback(self)

    def __str__(self):
        return '{}: {}'.format(self._name, self._array)

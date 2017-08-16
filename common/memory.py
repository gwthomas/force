import numpy as np


class Memory:
    def __init__(self, capacity):
        self._capacity = capacity
        self._array = []

    def __len__(self):
        return len(self._array)

    def __getitem__(self, i):
        return self._array[i]

    def clear(self):
        self._array = []

    def add(self, o):
        self._array.append(o)
        if len(self._array) > self._capacity:
            # Forget oldest
            self._array.pop(0)

    def sample(self, n, replace=True):
        indices = np.random.choice(len(self._array), size=n, replace=replace)
        return [self._array[i] for i in indices]

    def recent(self, n):
        return self._array[-n:]

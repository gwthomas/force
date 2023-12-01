import torch

from force.data.base import AbstractData, Dataset
from force.nn.util import get_device, torchify


class CircularData(AbstractData):
    def __init__(self, dtype: torch.dtype, elem_shape: tuple, capacity: int,
                 device=None):
        super().__init__(dtype)
        assert isinstance(elem_shape, tuple)
        for dim in elem_shape:
            assert isinstance(dim, int)
        device = get_device(device)

        self._dtype = dtype
        self._elem_shape = elem_shape
        self._capacity = capacity
        self._device = device

        buf_shape = (capacity, *elem_shape)
        self._buf = torch.empty(*buf_shape, dtype=dtype, device=device)
        self._counter = 0

    def __len__(self):
        return min(self._counter, self._capacity)

    @property
    def capacity(self):
        return self._capacity

    def _shape(self):
        return (len(self), *self._elem_shape)

    def _get(self, indices):
        assert torch.all(indices < len(self))
        if self._counter <= self._capacity:
            return self._buf[indices]
        else:
            offset = self._counter % self._capacity
            return self._buf[indices - offset]

    def append(self, x):
        x = torchify(x, device=self._device)
        assert x.shape == self._elem_shape
        i = self._counter % self._capacity
        self._buf[i] = x
        self._counter += 1

    def extend(self, xs):
        assert xs.shape[1:] == self._elem_shape
        batch_size = len(xs)
        assert batch_size <= self._capacity, 'Trying to extend by more than buffer capacity'
        i = self._counter % self._capacity
        end = i + batch_size
        if end <= self.capacity:
            self._buf[i:end] = xs
        else:
            fit = self.capacity - i
            overflow = end - self.capacity
            # Note: fit + overflow = batch_size
            self._buf[-fit:] = xs[:fit]
            self._buf[:overflow] = xs[-overflow:]
        self._counter += batch_size


class CircularDataset(Dataset):
    def __init__(self, components: dict, capacity: int, device=None):
        super().__init__({
            k: CircularData(dtype, shape, capacity, device)
            for k, (dtype, shape) in components.items()
        }, device=device)
        self._capacity = capacity

    @property
    def capacity(self):
        return self._capacity

    def _check_keys(self, items: dict):
        expected_keyset = set(self.keys())
        actual_keyset = set(items.keys())
        assert actual_keyset == expected_keyset, \
            f'Expected keys {expected_keyset}, but received {actual_keyset}'

    def append(self, items: dict):
        self._check_keys(items)
        for k, x in items.items():
            self._data[k].append(x)

    def extend(self, items: dict):
        self._check_keys(items)
        k0 = self.keys()[0]
        for k, xs in items.items():
            assert len(xs) == len(items[k0]), 'All data must have same length'
            self._data[k].extend(xs)
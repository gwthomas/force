from abc import ABC, abstractmethod

import h5py
import torch

from force.nn.shape import is_valid_shape
from force.nn.util import TORCH_INT_TYPES, get_device, torchify, random_indices


# Converts various index objects into a standard form (a torch.Tensor of int type)
def _get_indices(item) -> torch.Tensor:
    if isinstance(item, torch.Tensor):
        assert item.dtype in TORCH_INT_TYPES, 'Indices must be integer type'
        assert item.ndim == 1, 'Indices must be 1-D'
        return item
    elif isinstance(item, int):
        return torch.tensor([item])
    elif isinstance(item, slice):
        step = 1 if item.step is None else item.step
        return torch.arange(item.start, item.stop, step)
    elif isinstance(item, list):
        return torch.tensor(item)
    else:
        raise KeyError(f'Invalid key: {item}')


class AbstractData(ABC):
    def __init__(self, dtype):
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @abstractmethod
    def _shape(self) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def _get(self, indices) -> torch.Tensor:
        raise NotImplementedError

    @property
    def shape(self):
        shape = self._shape()
        assert is_valid_shape(shape)
        return shape

    def __getitem__(self, indices: torch.Tensor) -> torch.Tensor:
        return self._get(indices)

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return f'<{type(self).__name__} type={self.dtype} shape={self.shape}>'

    def get(self) -> torch.Tensor:
        return self._get(torch.arange(len(self)))


class Dataset:
    def __init__(self, data: dict, device=None):
        self._data = data
        self._device = device

    def __len__(self):
        lengths = [len(v) for v in self.values()]
        # Lengths should all be the same
        for l in lengths:
            assert l == lengths[0]
        return lengths[0]

    def __repr__(self):
        rep = [f'<Dataset of length {len(self)}, components:']
        for k, v in self.items():
            rep.append(f'\n\t{k}: {v}')
        rep.append('>')
        return ''.join(rep)

    @property
    def device(self):
        return self._device

    def keys(self):
        return tuple(self._data.keys())

    def values(self):
        return tuple(self._data.values())

    def items(self):
        return tuple(self._data.items())

    def dtypes(self):
        return {k: v.dtype for k, v in self._data.items()}

    def shapes(self):
        return {k: v.shape for k, v in self._data.items()}

    def __getattr__(self, attr):
        if attr in self._data:
            return self._data[attr]
        else:
            raise AttributeError

    def __getitem__(self, item) -> dict:
        if isinstance(item, int):
            # Select the same index from each component
            indices = _get_indices(item)
            return {
                k: v[indices][0].to(self._device) for k, v in self._data.items()
            }
        elif isinstance(item, torch.Tensor) or type(item) in {slice, list}:
            # Select the same indices from each component
            indices = _get_indices(item)
            return {
                k: v[indices].to(self._device) for k, v in self._data.items()
            }
        else:
            raise KeyError(f'Invalid key: {item}')

    def sample(self, n, replace=False) -> dict:
        indices = random_indices(len(self), size=n, replace=replace)
        return self[indices]

    def get(self, *args, as_dict=False):
        keys = self.keys() if len(args) == 0 else args
        values = [self._data[k].get().to(self._device) for k in keys]
        if as_dict:
            return dict(zip(keys, values))
        elif len(args) == 1:
            return values[0]
        else:
            return values

    def save(self, path):
        with h5py.File(path, 'w') as f:
            for k, v in self.items():
                f.create_dataset(k, data=v.get().numpy())


class TensorData(AbstractData):
    def __init__(self, tensor: torch.Tensor):
        self._buf = tensor

    def _dtype(self):
        return self._buf.dtype

    def _shape(self):
        return tuple(self._buf.shape)

    def _get(self, indices):
        return self._buf[indices]

class TensorDataset(Dataset):
    def __init__(self, data: dict, device=None):
        device = get_device(device)
        super().__init__({
            k: TensorData(v.to(device)) for k, v in data.items()
        }, device=device)
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

    @property
    def shape(self):
        shape = self._shape()
        assert is_valid_shape(shape), f'{shape} is not valid shape'
        return shape

    @abstractmethod
    def _get(self, indices) -> torch.Tensor:
        raise NotImplementedError

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
        self._data = dict(data)
        self._device = get_device(device)

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

    # def __getattr__(self, attr):
    #     if attr in self._data:
    #         return self._data[attr]
    #     else:
    #         raise AttributeError

    def __getitem__(self, item) -> dict:
        indices = _get_indices(item)
        ret = {}
        for k, v in self._data.items():
            ret[k] = v[indices].to(self._device)
            if isinstance(item, int):
                ret[k] = ret[k][0]
        return ret

    def sample(self, n, replace=False) -> dict:
        indices = random_indices(len(self), size=n, replace=replace, device=self._device)
        return self[indices]

    def get(self, *args, as_dict=False, device=None):
        keys = self.keys() if len(args) == 0 else args
        device = self._device if device is None else device
        values = [self._data[k].get().to(device) for k in keys]
        if as_dict:
            return dict(zip(keys, values))
        elif len(args) == 1:
            return values[0]
        else:
            return values

    def to(self, device: torch.device):
        for v in self._data.values():
            v.to(device)
        self._device = device

    def save_to_file(self, f: h5py.File, prefix: str = None):
        assert isinstance(f, h5py.File)
        for k, v in self.items():
            dataset_name = k if prefix is None else f'{prefix}/{k}'
            f.create_dataset(dataset_name, data=v.get().cpu().numpy())

    def save_to_path(self, path: str, prefix: str = None):
        with h5py.File(path, 'w') as f:
            self.save_to_file(f, prefix)


class TensorData(AbstractData):
    def __init__(self, tensor: torch.Tensor, copy=True):
        super().__init__(tensor.dtype)
        self._buf = tensor.clone() if copy else tensor
        self._device = tensor.device

    def _shape(self):
        return self._buf.shape

    def _get(self, indices):
        return self._buf[indices]

    def __repr__(self):
        return f'<{type(self).__name__} type={self.dtype} shape={self.shape} device={self._device}>'

    def to(self, device: torch.device):
        self._buf = self._buf.to(device)
        self._device = device

    def share_memory(self):
        self._buf.share_memory_()

    @staticmethod
    def from_data(data: AbstractData):
        return TensorData(data.get())

class TensorDataset(Dataset):
    def __init__(self, data: dict, copy=True, device=None):
        device = get_device(device)
        super().__init__({
            k: TensorData(v.to(device), copy) for k, v in data.items()
        }, device=device)

    def share_memory(self):
        for v in self._data.values():
            v.share_memory()

    @staticmethod
    def from_dataset(dataset: Dataset, copy=True):
        return TensorDataset(dataset.get(as_dict=True), copy, dataset.device)
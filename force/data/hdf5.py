from pathlib import Path

import h5py
import numpy as np
import torch

from force.data.base import AbstractData, Dataset, TensorData
from force.nn.util import random_indices


def _torch_dtype_for_dataset(dataset: h5py.Dataset):
    assert isinstance(dataset, h5py.Dataset)
    return torch.from_numpy(np.zeros([], dtype=dataset.dtype)).dtype

def _tensor_for_dataset(dataset: h5py.Dataset):
    assert isinstance(dataset, h5py.Dataset)
    return torch.from_numpy(np.array(dataset))


class HDF5BackedData(AbstractData):
    def __init__(self, dataset: h5py.Dataset):
        dtype = _torch_dtype_for_dataset(dataset)
        super().__init__(dtype)
        self._dataset = dataset

    def _shape(self):
        return tuple(self._dataset.shape)

    def _get(self, indices):
        return _tensor_for_dataset(self._dataset[indices.numpy()])


class HDF5BackedDataset(Dataset):
    def __init__(self, path: str, device=None):
        self._path = Path(path)
        assert self._path.is_file(), f'No file found at path {self._path}'
        self._file = h5py.File(self._path, 'r')

        super().__init__({
            k: HDF5BackedData(v) for k, v in self._file.items()
        }, device=device)

    @property
    def using_file(self):
        return self._file is not None

    def cache(self, *args):
        keys = self.keys() if len(args) == 0 else args
        for k in keys:
            self._data[k] = TensorData(self._data[k].get().to(self._device))

        if all(isinstance(v, TensorData) for v in self.values()):
            # All data has been read into memory, so no need to keep file open
            self.close_if_open()

    def close(self):
        assert self.using_file, 'No file to close'
        self._file.close()
        self._file = None

    def close_if_open(self):
        if self.using_file:
            self.close()

    # Override because h5py expects indices in increasing order
    def sample(self, n, replace=False):
        indices = random_indices(len(self), size=n, replace=replace)
        if self.using_file:
            indices = sorted(indices)
        return self[indices]
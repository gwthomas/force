import math

from frozendict import frozendict
import torch
import torch.nn as nn
import torch.nn.functional as F

from force.nn.module import Module
from force.nn.shape import shape2str


def cat_dims(input_shape, dim=-1):
    assert dim == -1, 'Currently only dim=-1 is supported, but that may change'
    if isinstance(input_shape, tuple):
        shape_list = list(input_shape)
    elif isinstance(input_shape, frozendict):
        shape_list = list(input_shape.values())
    else:
        raise ValueError('cat_dims expects complex shape (tuple or frozendict of Sizes)')
    other_dims = shape_list[0][:-1]
    total_dim = 0
    for shape in shape_list:
        assert shape[:-1] == other_dims
        total_dim += shape[-1]
    return torch.Size([*other_dims, total_dim])


class Cat(Module):
    """Concatenates inputs along their final dimension"""
    def __init__(self, input_shape, dim=-1):
        assert type(input_shape) in {tuple, frozendict}
        super().__init__()
        self._input_shape = input_shape
        self._output_shape = cat_dims(input_shape, dim=dim)
        self.dim = dim

    @property
    def output_dim(self):
        assert len(self._output_shape) == 1
        return self._output_shape[0]

    def forward(self, input, **kwargs):
        if type(input) in {list, tuple}:
            return torch.cat(input, dim=self.dim)
        elif isinstance(input, dict):
            return torch.cat(input.values(), dim=self.dim)
        else:
            raise ValueError(f'Cannot cat {input}')

    def extra_repr(self):
        return f'{shape2str(self._input_shape)} -> {shape2str(self._output_shape)}'


class Split(Module):
    """Splits input along its final dimension"""
    def __init__(self, output_shape, dim=-1):
        super().__init__()
        self._input_shape = cat_dims(output_shape, dim=dim)
        self._output_shape = output_shape
        self.dim = dim
        if isinstance(output_shape, tuple):
            self._split_sizes = [s[dim] for s in output_shape]
        elif isinstance(output_shape, frozendict):
            self._split_sizes = [s[dim] for s in output_shape.values()]
        else:
            raise ValueError('Split output shape should be tuple or frozendict')

    def forward(self, input, **kwargs):
        split = input.split(self._split_sizes, dim=self.dim)
        if isinstance(self._output_shape, tuple):
            return split
        elif isinstance(self._output_shape, frozendict):
            return dict(zip(self._output_shape.keys(), split))
        else:
            raise ValueError('Split output shape should be tuple or frozendict')

    def extra_repr(self):
        return f'{shape2str(self._input_shape)} -> {shape2str(self._output_shape)}'


class Linear(Module, nn.Linear):
    def __init__(self, *args, **kwargs):
        Module.__init__(self, super_init=False)
        nn.Linear.__init__(self, *args, **kwargs)

        self._input_shape = torch.Size([self.in_features])
        self._output_shape = torch.Size([self.out_features])

    def forward(self, input, **kwargs):
        return nn.Linear.forward(self, input)



def odd_log(x):
    """An unbounded sigmoid-like activation: sgn(x)log(1+|x|)"""
    s = torch.sign(x)
    return s * torch.log(1 + s*x)

NAMED_POINTWISE_ACTIVATIONS = {
    'relu': F.relu,
    'softplus': F.softplus,
    'silu': F.silu,
    'swish': F.silu,    # alias
    'sigmoid': F.sigmoid,
    'tanh': F.tanh,
    'oddlog': odd_log
}

class PointwiseActivation(Module):
    def __init__(self, name):
        super().__init__()
        assert name in NAMED_POINTWISE_ACTIVATIONS
        self.name = name
        self.fn = NAMED_POINTWISE_ACTIVATIONS[name]

    def get_output_shape(self, input_shape, **kwargs):
        return input_shape

    def forward(self, x, **kwargs):
        return self.fn(x)

    def extra_repr(self):
        return self.name


class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def get_output_shape(self, input_shape, **kwargs):
        return input_shape

    def forward(self, x, **kwargs):
        return F.softmax(x, dim=self.dim)

    def extra_repr(self):
        return f'dim={self.dim}'


class Squeeze(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def get_output_shape(self, input_shape, **kwargs):
        assert input_shape[self.dim] == 1
        return input_shape[:self.dim]

    def forward(self, x, **kwargs):
        return torch.squeeze(x, dim=self.dim)

    def extra_repr(self):
        return f'dim={self.dim}'
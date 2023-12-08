import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from force.nn.module import Module
from force.nn.util import get_device


class Normalizer(Module):
    def __init__(self, dim, device=None):
        super().__init__()
        self.dim = dim
        self._device = device = get_device(device)
        self.register_buffer('mean', torch.zeros(dim, device=device))
        self.register_buffer('std', torch.ones(dim, device=device))

    def fit(self, x):
        assert x.ndim == 2
        assert x.shape[1] == self.dim
        std, mean = torch.std_mean(x, dim=0)
        self.mean.copy_(mean)
        self.std.copy_(std)

    def get_output_shape(self, input_shape):
        assert input_shape[-1] == self.dim
        return input_shape

    def forward(self, x):
        return (x - self.mean) / self.std

    def extra_repr(self):
        return f'dim={self.dim}'


def cat_dims(input_shape):
    shape_list = list(input_shape) if isinstance(input_shape, tuple) \
        else list(input_shape.values)
    other_dims = shape_list[0][:-1]
    total_dim = 0
    for shape in shape_list:
        assert shape[:-1] == other_dims
        total_dim += shape[-1]
    return torch.Size([*other_dims, total_dim])


class Cat(Module):
    """Concatenates inputs along their final dimension
    """
    def __init__(self, input_dims):
        assert type(input_dims) in {list, tuple}
        super().__init__()
        self._input_dims = tuple(input_dims)
        self._output_dim = sum(input_dims)

    @property
    def output_dim(self):
        return self._output_dim

    def get_output_shape(self, input_shape, **kwargs):
        assert len(input_shape) == len(self._input_dims)
        for d, s in zip(self._input_dims, input_shape):
            assert s[-1] == d
        return cat_dims(input_shape)

    def forward(self, input, **kwargs):
        if type(input) in {list, tuple}:
            return torch.cat(input, dim=-1)
        elif isinstance(input, dict):
            return torch.cat(input.values(), dim=-1)
        else:
            raise ValueError(f'Cannot cat {input}')

    def extra_repr(self):
        return f'{self._input_dims} -> {self._output_dim}'


class Split(Module):
    def __init__(self, output_shape, dim):
        super().__init__()
        self._input_shape = cat_dims(output_shape)
        self._output_shape = output_shape
        self.dim = dim

    def get_output_shape(self, input_shape, **kwargs):
        self.verify_input_shape(input_shape)
        return self._output_shape

    def forward(self, input, **kwargs):
        # TODO
        breakpoint()


class Linear(Module, nn.Linear):
    def __init__(self, *args, **kwargs):
        Module.__init__(self, super_init=False)
        nn.Linear.__init__(self, *args, **kwargs)

    def get_output_shape(self, input_shape, **kwargs):
        assert input_shape[-1] == self.in_features
        return torch.Size([*input_shape[:-1], self.out_features])

    def forward(self, input, **kwargs):
        return nn.Linear.forward(self, input)



def odd_log(x):
    """An unbounded sigmoid-like activation"""
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

    def forward(self, x):
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


class LinearEnsemble(Module):
    """Represents an ensemble of linear layers. The forward pass
    efficiently computes all layers' outputs via batched matrix multiplication.
    """

    shape_relevant_kwarg_keys = {'num_models'}

    def __init__(self, num_models: int, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearEnsemble, self).__init__()
        self.num_models = num_models
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((num_models, out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_models, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        for i in range(self.num_models):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[i], -bound, bound)

    def get_output_shape(self, input_shape, **kwargs):
        assert isinstance(input_shape, torch.Size)
        num_models = kwargs['num_models']
        assert num_models <= self.num_models
        if len(input_shape) == 1:
            expected_shape = (self.in_features,)
        elif len(input_shape) == 2:
            expected_shape = (num_models, self.in_features)
        else:
            raise ValueError(f'Invalid input shape to LinearEnsemble: {input_shape}')
        if input_shape != expected_shape:
            raise ValueError(f'Expected shape {expected_shape} but got shape {input_shape}')
        return torch.Size([num_models, self.out_features])

    def forward(self, input, **kwargs):
        num_models = kwargs['num_models']
        if num_models is None or num_models == self.num_models:
            num_models = self.num_models
            W = self.weight
            b = self.bias
        else:
            assert num_models <= self.num_models
            model_indices = kwargs['model_indices']
            assert model_indices is not None
            W = self.weight[model_indices]
            b = self.bias[model_indices]

        batch_size = input.shape[0]
        ndim = input.dim()
        assert ndim in {2, 3}
        if ndim == 2:
            input = input.unsqueeze(1).tile(1, num_models, 1)
        assert input.shape == (batch_size, num_models, self.in_features)

        # Tile weights and reshape for batched matrix-vector multiplication
        W = W.unsqueeze(0).tile(batch_size, 1, 1, 1)
        W = W.reshape(-1, self.out_features, self.in_features)
        input = input.reshape(-1, self.in_features, 1)
        Wx = torch.bmm(W, input).squeeze(-1)
        Wx = Wx.reshape(batch_size, num_models, self.out_features)
        out = Wx + b

        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
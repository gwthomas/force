from abc import ABC, abstractmethod

from frozendict import frozendict
import torch
import torch.nn as nn

from force.config import Configurable
from force.nn.shape import is_valid_shape, get_nonbatch_shape, matches_shape, shape2str


class Module(nn.Module):
    # If the output shape of the module depends on any optional arguments,
    # which are passed as kwargs, list the keys here
    shape_relevant_kwarg_keys = set()

    def __init__(self, super_init=True):
        if super_init:
            super().__init__()
        self._device = None

        # If this module can only accept inputs of one particular shape, set _input_shape.
        self._input_shape = None
        self._output_shape = None

        # Maps (input shape, shape-relevant kwargs) -> output shape
        self._shape_cache = {}

    @property
    def device(self):
        return self._device

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        if self._output_shape is None:
            assert self.input_shape is not None
            self._output_shape = self.get_output_shape(self._input_shape)
        return self._output_shape

    def verify_input_shape(self, input_shape):
        assert self._input_shape is not None
        assert input_shape == self._input_shape, \
            f'{self.__class__.__name__} expected {shape2str(self._input_shape)} but got {shape2str(input_shape)}'

    def verify_input(self, input, batch_dims):
        assert self._input_shape is not None
        assert matches_shape(input, self._input_shape, batch_dims)

    @abstractmethod
    def get_output_shape(self, input_shape, **kwargs):
        raise NotImplementedError(f'{self} does not implement get_output_shape')

    def __call__(self, input, **kwargs):
        batch_dims = kwargs.get('batch_dims', 1)
        input_shape = get_nonbatch_shape(input, batch_dims)

        if self.input_shape is not None:
            self.verify_input_shape(input_shape)
            self.verify_input(input, batch_dims)

        shape_relevant_kwargs = frozendict({
            k: v for k, v in kwargs.items() if k in self.shape_relevant_kwarg_keys
        })

        # Check shape compatibility
        cache_key = (input_shape, shape_relevant_kwargs)
        if cache_key in self._shape_cache:
            expected_output_shape = self._shape_cache[cache_key]
        else:
            expected_output_shape = self.get_output_shape(input_shape, **kwargs)
            assert is_valid_shape(expected_output_shape)
            self._shape_cache[cache_key] = expected_output_shape

        output = self.forward(input, **kwargs)
        # Check output shape
        assert matches_shape(output, expected_output_shape), \
               f'{self.__class__.__name__}: ' + \
               f'Expected output shape {shape2str(expected_output_shape)}, ' + \
               f'but got {shape2str(get_nonbatch_shape(output, 1))}. ' + \
               f'(Did you specify shape_relevant_kwarg_keys?)'
        return output

    def to(self, device):
        super().to(device)
        self._device = device
        for m in self.modules():
            # if hasattr(m, '_device'):
            m._device = device
        return self

    def cpu(self):
        self.to(torch.device('cpu'))

    def cuda(self, device=None):
        self.to(torch.device('cuda' if device is None else device))

    def save(self, path, prefix='', keep_vars=False):
        state_dict = self.state_dict(prefix=prefix, keep_vars=keep_vars)
        torch.save(state_dict, path)

    def load(self, path, strict=True):
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict, strict=strict)

    def set_requires_grad(self, requires_grad):
        for p in self.parameters():
            p.requires_grad = requires_grad

    def count_parameters(self, only_requires_grad=False):
        total = 0
        for p in self.parameters():
            if only_requires_grad and not p.requires_grad:
                continue
            total += p.numel()
        return total


class ModuleList(Module, nn.ModuleList):
    def __init__(self, modules=None):
        Module.__init__(self, super_init=False)
        nn.ModuleList.__init__(self, modules=modules)

    def get_output_shape(self, input_shape, **kwargs):
        raise NotImplementedError


class Sequential(Module, nn.Sequential):
    def __init__(self, *args):
        Module.__init__(self, super_init=False)
        nn.Sequential.__init__(self, *args)

    def get_output_shape(self, input_shape, **kwargs):
        shape = input_shape
        for layer in self:
            shape = layer.get_output_shape(shape, **kwargs)
        return shape

    def forward(self, input, **kwargs):
        x = input
        for layer in self:
            x = layer(x, **kwargs)
        return x


class ConfigurableModule(Configurable, Module):
    """A convenience class that simply combines two very common superclasses"""
    def __init__(self, cfg):
        Configurable.__init__(self, cfg)
        Module.__init__(self)
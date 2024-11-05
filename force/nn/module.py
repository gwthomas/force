from frozendict import frozendict
import torch
import torch.nn as nn

from force.defaults import DEVICE
from force.config import Configurable
from force.nn.shape import is_valid_shape, get_nonbatch_shape, matches_shape, shape2str
from force.nn.util import torchify


class Module(nn.Module):
    # If the output shape of the module depends on any optional arguments,
    # which are passed as kwargs, list the keys here
    shape_relevant_kwarg_keys = set()

    def __init__(self, device=None, super_init=True):
        if super_init:
            super().__init__()

        self._device = DEVICE if device is None else device

        # If this module can only accept inputs of one particular shape, set self._input_shape
        # and optionally set self._output_shape instead of get_output_shape
        self._input_shape = None
        self._output_shape = None

        # Maps (input shape, shape-relevant kwargs) -> output shape
        self._shape_cache = {}

    @property
    def device(self):
        return self._device

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, Module) and self._device is not None:
            value.to(self._device)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def verify_input_shape(self, input_shape):
        assert self._input_shape is not None
        assert input_shape == self._input_shape, \
            f'{self.__class__.__name__} expected {shape2str(self._input_shape)} but got {shape2str(input_shape)}'

    def get_output_shape(self, input_shape, **kwargs):
        if self._input_shape is not None and self._output_shape is not None:
            self.verify_input_shape(input_shape)
            return self._output_shape
        else:
            # Override this method to implement custom logic
            raise NotImplementedError(f'{self} does not implement get_output_shape')

    def __call__(self, input, **kwargs):
        batch_dims = kwargs.get('batch_dims', 1)
        input_shape = get_nonbatch_shape(input, batch_dims)

        if self._input_shape is not None:
            self.verify_input_shape(input_shape)

        if self._output_shape is not None:
            expected_output_shape = self._output_shape
        else:
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

        input = torchify(input, device=self.device)
        output = self.forward(input, **kwargs)

        # Check output shape
        assert matches_shape(output, expected_output_shape), \
               f'{self.__class__.__name__}: ' + \
               f'Expected output shape {shape2str(expected_output_shape)}, ' + \
               f'but got {shape2str(get_nonbatch_shape(output, batch_dims))}. ' + \
               f'(Did you specify shape_relevant_kwarg_keys?)'

        return output

    def to(self, device):
        super().to(device)
        self._device = device
        for m in self.modules():
            if isinstance(m, Module):
                m._device = device
        return self

    def cpu(self):
        self.to(torch.device('cpu'))

    def cuda(self):
        self.to(torch.device('cuda'))

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


class Sequential(Module, nn.Sequential):
    def __init__(self, *args, device=None):
        Module.__init__(self, device=device, super_init=False)
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
    def __init__(self, cfg, device=None):
        Module.__init__(self, device=device)
        Configurable.__init__(self, cfg)

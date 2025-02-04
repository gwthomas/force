from typing import Callable

import torch
import torch.export
import torch.nn as nn

from force.defaults import DEVICE
from force.config import Configurable
from force.nn.shape import Shape, shape2str


class Module(nn.Module):
    def __init__(self, device=None, super_init=True):
        if super_init:
            super().__init__()

        self._device = DEVICE if device is None else device

        # If this module can only accept inputs of one particular shape, set self._input_shape
        # and optionally set self._output_shape instead of get_output_shape
        self._input_shape = None
        self._output_shape = None

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
    
    def export(self, path: str | None = None) -> torch.export.ExportedProgram:
        input_shape = self.input_shape
        assert input_shape is not None, \
            "Cannot export Module without input_shape"
        
        assert isinstance(input_shape, torch.Size), \
            "Currently only tensor inputs are supported"
        example_input = torch.zeros(2, *input_shape, device=self.device)
        example_args = (example_input,)
        
        # Dynamic batch size
        dynamic_shapes = ({0: torch.export.Dim("batch")},)

        program = torch.export.export(
            self, args=example_args, dynamic_shapes=dynamic_shapes
        )
        if path is not None:
            torch.export.save(program, path)
        return program


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


class LambdaModule(Module):
    def __init__(self, fn: Callable,
                 named_parameters: dict | None = None,
                 input_shape: Shape | None = None):
        super().__init__()
        self.fn = fn

        if named_parameters is not None:
            for name, param in named_parameters.items():
                # PyTorch won't allow registration of a name containing dots,
                # but Module.named_parameters() returns names containing dots.
                name = name.replace('.', '_')
                self.register_parameter(name, param)
        
        if input_shape is not None:
            self._input_shape = input_shape

    def forward(self, input):
        return self.fn(input)

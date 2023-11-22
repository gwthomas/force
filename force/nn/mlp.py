from frozendict import frozendict

import torch
from force import defaults
from force.config import BaseConfig, Configurable, Field
from force.nn.module import Sequential
from force.nn.layers import Cat, Linear, LinearEnsemble, PointwiseActivation, Split, Squeeze
from force.nn.shape import is_valid_shape, shape_numel


def ge1(x):
    return x >= 1

class MLP(Configurable, Sequential):
    class Config(BaseConfig):
        num_hidden_layers = Field(defaults.HIDDEN_LAYERS, check=ge1)
        hidden_dim = Field(defaults.HIDDEN_DIM, check=ge1)
        activation = defaults.ACTIVATION
        use_bias = True

    def __init__(self, cfg, input_shape, output_shape=None,
                 final_activation=None):
        assert is_valid_shape(input_shape)
        assert is_valid_shape(output_shape) or output_shape is None
        Configurable.__init__(self, cfg)
        Sequential.__init__(self)

        # Construct layers
        layers = []
        if isinstance(input_shape, torch.Size):
            prev_dim = input_shape.numel()
        else:
            if isinstance(input_shape, tuple):
                input_dims = [s[-1] for s in input_shape]
            elif isinstance(input_shape, frozendict):
                input_dims = [s[-1] for s in input_shape.values()]
            else:
                raise ValueError(f'Invalid shape: {input_shape}')
            cat_layer = Cat(input_dims)
            layers.append(cat_layer)
            prev_dim = cat_layer.output_dim
        for _ in range(cfg.num_hidden_layers):
            layers.append(self._new_layer(prev_dim, cfg.hidden_dim))
            layers.append(PointwiseActivation(cfg.activation))
            prev_dim = cfg.hidden_dim

        # (optional) final layer and activation
        if output_shape is not None:
            output_numel = shape_numel(output_shape)
            layers.append(self._new_layer(prev_dim, output_numel))
        if final_activation is not None:
            layers.append(PointwiseActivation(final_activation))

        # (optional) remove last dimension if it's a scalar
        if output_shape == torch.Size([]):
            layers.append(Squeeze())

        # Add split if needed
        if type(output_shape) in {tuple, frozendict}:
            layers.append(Split(output_shape))

        Sequential.__init__(self, *layers)

    def _new_layer(self, in_dim, out_dim):
        return Linear(in_dim, out_dim, bias=self.cfg.use_bias)


class MLPEnsemble(MLP):
    """Represents an ensemble of feedforward networks. The forward pass
    efficiently computes all models' outputs via batched matrix multiplication.
    """

    shape_relevant_kwarg_keys = {'num_models'}

    class Config(MLP.Config):
        num_models = Field(int)

    def _new_layer(self, in_dim, out_dim):
        return LinearEnsemble(self.cfg.num_models, in_dim, out_dim, bias=self.cfg.use_bias)
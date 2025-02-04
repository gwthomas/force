import torch
from force import defaults
from force.config import BaseConfig, Configurable, Field
from force.nn.module import Sequential
from force.nn.layers import Cat, Linear, PointwiseActivation, Split, Squeeze
from force.nn.shape import is_valid_shape, shape_numel


class MLP(Configurable, Sequential):
    class Config(BaseConfig):
        num_hidden_layers = defaults.HIDDEN_LAYERS
        hidden_dim = defaults.HIDDEN_DIM
        activation = defaults.ACTIVATION
        use_bias = True

    def __init__(self, cfg, input_shape, output_shape=None,
                 final_activation=None):
        assert cfg.num_hidden_layers >= 1
        assert cfg.hidden_dim >= 1
        assert is_valid_shape(input_shape)
        assert is_valid_shape(output_shape) or output_shape is None
        Configurable.__init__(self, cfg)
        Sequential.__init__(self)

        self._input_shape = input_shape
        self._output_shape = output_shape if output_shape is not None \
                             else torch.Size([cfg.num_hidden_layers])

        layers = []

        if isinstance(input_shape, torch.Size):
            prev_dim = input_shape.numel()
        else:
            # Input is of complex shape; add initial concatenation layer
            cat_layer = Cat(input_shape)
            layers.append(cat_layer)
            prev_dim = cat_layer.output_dim

        # Hidden layers
        for _ in range(cfg.num_hidden_layers):
            layers.append(self._new_layer(prev_dim, cfg.hidden_dim))
            layers.append(PointwiseActivation(cfg.activation))
            prev_dim = cfg.hidden_dim

        # Posibly add final layer and activation
        if output_shape is not None:
            output_numel = shape_numel(output_shape)
            layers.append(self._new_layer(prev_dim, output_numel))
        if final_activation is not None:
            layers.append(PointwiseActivation(final_activation))

        # Possibly add layer that reshapes output
        if output_shape == torch.Size([]):
            layers.append(Squeeze())
        elif type(output_shape) in {list, dict}:
            layers.append(Split(output_shape))

        Sequential.__init__(self, *layers)

    def _new_layer(self, in_dim, out_dim):
        return Linear(in_dim, out_dim, bias=self.cfg.use_bias)
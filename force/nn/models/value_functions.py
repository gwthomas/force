import random

import torch
import torch.nn as nn

from force.nn import ConfigurableModule
from force.nn.models import MLP


def _get_input_shape(obs_shape, action_shape, goal_shape):
    input_shape = [obs_shape]
    if action_shape is not None:
        input_shape.append(action_shape)
    if goal_shape is not None:
        input_shape.append(goal_shape)
    if len(input_shape) == 1:
        return input_shape[0]
    else:
        return input_shape


class GeneralValueFunction(ConfigurableModule):
    """Base class for a single value/Q function.
    Maps an observation, and optionally an action and/or goal, to a value.
    """

    class Config(ConfigurableModule.Config):
        net = MLP.Config

    def __init__(self, cfg, obs_shape, action_shape=None, goal_shape=None,
                 normalizer=None):
        super().__init__(cfg)
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.goal_shape = goal_shape
        self.normalizer = normalizer

        self._input_shape = _get_input_shape(obs_shape, action_shape, goal_shape)
        self._output_shape = torch.Size([])   # scalar
        self.net = MLP(cfg.net, self._input_shape, self._output_shape)

    @property
    def is_action_conditioned(self):
        return self.action_shape is not None

    @property
    def is_goal_conditioned(self):
        return self.goal_shape is not None

    def forward(self, obs, **kwargs):
        if self.normalizer is not None:
            obs = self.normalizer(obs)
        return self.net(obs)


class GeneralValueFunctionEnsemble(ConfigurableModule):
    """Base class for value/Q function ensembles.
    Maps an observation, and optionally an action and/or goal, to a value.
    """

    class Config(ConfigurableModule.Config):
        net = MLP.Config
        num_models = int

    def __init__(self, cfg, obs_shape, action_shape=None, goal_shape=None,
                 normalizer=None):
        super().__init__(cfg)
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.goal_shape = goal_shape
        self.normalizer = normalizer

        self._input_shape = _get_input_shape(obs_shape, action_shape, goal_shape)
        output_shape = torch.Size([])   # scalar
        self.nets = nn.ModuleList([
            MLP(cfg.net, self._input_shape, output_shape) for _ in range(cfg.num_models)
        ])
        self.num_models = cfg.num_models

    @property
    def is_action_conditioned(self):
        return self.action_shape is not None

    @property
    def is_goal_conditioned(self):
        return self.goal_shape is not None

    def _process_which(self, which, **kwargs):
        if isinstance(which, tuple):
            num_models = len(which)
            model_indices = which
        elif which == 'random':
            num_models = 1
            model_indices = [random.randrange(self.num_models)]
        else:
            # all will be used
            num_models = self.num_models
            # model_indices = None
            model_indices = list(range(num_models))
        return dict(kwargs, num_models=num_models, model_indices=model_indices)

    def get_output_shape(self, input_shape, **kwargs):
        self.verify_input_shape(input_shape)
        which = kwargs.pop('which')
        new_kwargs = self._process_which(which)
        output_shape = torch.Size([new_kwargs['num_models']])
        if isinstance(which, tuple) or which == 'all':
            return output_shape
        elif which in {'mean', 'min', 'random'}:
            # First dimension (which indexes into models) will be reduced
            return output_shape[1:]
        else:
            raise ValueError(f'Invalid ensemble option: {which}')

    def forward(self, input, **kwargs):
        """
        The `which` argument should be one of the following:
          * a tuple of indices to directly specify which model(s) to use
          * "random" to randomly pick a model from the ensemble
          * "all" to get the outputs of all models in the ensemble
          * "min" to compute the minimum of all models in the ensemble
          * "mean" to compute the average of all models in the ensemble
        """
        if self.normalizer is not None:
            input = self.normalizer(input)

        # Select which model(s) to use
        which = kwargs.pop('which')
        new_kwargs = self._process_which(which, **kwargs)

        # Batched forward passes
        # out = self.ensemble(input, **new_kwargs)
        out = torch.stack(
            [self.nets[i](input) for i in new_kwargs['model_indices']],
            dim=1
        )

        # Reduction
        if which == 'random':
            out = out[:, 0]
        elif which == 'min':
            out = out.min(1).values
        elif which == 'mean':
            out = out.mean(1)
        elif isinstance(which, tuple) or which == 'all':
            pass
        else:
            raise ValueError(f'Invalid ensemble option: {which}')
        return out


# Special case: V(s) does not condition on action/goal
class ValueFunction(GeneralValueFunction):
    def __init__(self, cfg, obs_shape):
        super().__init__(cfg, obs_shape, action_shape=None, goal_shape=None)

class ValueFunctionEnsemble(GeneralValueFunctionEnsemble):
    def __init__(self, cfg, obs_shape):
        super().__init__(cfg, obs_shape, action_shape=None, goal_shape=None)


# Special case: Q(s,a) does not condition on goal
class QFunction(GeneralValueFunction):
    def __init__(self, cfg, obs_shape, action_shape):
        super().__init__(cfg, obs_shape, action_shape=action_shape, goal_shape=None)

class QFunctionEnsemble(GeneralValueFunctionEnsemble):
    def __init__(self, cfg, obs_shape, action_shape):
        super().__init__(cfg, obs_shape, action_shape=action_shape, goal_shape=None)


# Goal-conditioned versions
class GoalConditionedValueFunction(GeneralValueFunction):
    def __init__(self, cfg, obs_shape, goal_shape):
        super().__init__(cfg, obs_shape, action_shape=None, goal_shape=goal_shape)

class GoalConditionedValueFunctionEnsemble(GeneralValueFunctionEnsemble):
    def __init__(self, config, obs_shape, goal_shape):
        super().__init__(config, obs_shape, action_shape=None, goal_shape=goal_shape)

class GoalConditionedQFunction(GeneralValueFunction):
    def __init__(self, config, obs_shape, action_shape, goal_shape):
        super().__init__(config, obs_shape, action_shape=action_shape, goal_shape=goal_shape)

class GoalConditionedQFunctionEnsemble(GeneralValueFunctionEnsemble):
    def __init__(self, config, obs_shape, action_shape, goal_shape):
        super().__init__(config, obs_shape, action_shape=action_shape, goal_shape=goal_shape)
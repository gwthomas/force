import random

import torch

from force.nn import ConfigurableModule, MLP, MLPEnsemble


class DiscreteQFunction(ConfigurableModule):
    """Maps an observation to all actions' Q values"""
    class Config(ConfigurableModule.Config):
        mlp = MLP.Config()

    def __init__(self, cfg, obs_shape, num_actions):
        super().__init__(cfg)
        output_shape = torch.Size([num_actions])
        self.q = MLP(cfg.mlp, obs_shape, output_shape)
        self._input_shape = obs_shape
        self.num_actions = num_actions

    def get_output_shape(self, input_shape):
        return self.q.get_output_shape(input_shape)

    def forward(self, state, **kwargs):
        return self.q(state)


class DiscreteQFunctionEnsemble(ConfigurableModule):
    """Ensemble of functions which map observation to all Q values"""
    class Config(ConfigurableModule.Config):
        ensemble = MLPEnsemble.Config()

    def __init__(self, cfg, obs_shape, num_actions):
        super().__init__(cfg)
        output_shape = torch.Size([num_actions])
        self.ensemble = MLPEnsemble(cfg.ensemble, obs_shape, output_shape)
        self._input_shape = obs_shape
        self.num_actions = num_actions
        self.num_models = cfg.ensemble.num_models

    def get_output_shape(self, input_shape):
        return self.ensemble.get_output_shape(input_shape, num_models=self.num_models)

    def forward(self, obs, **kwargs):
        return self.ensemble(obs, num_models=self.num_models)


def _get_input_shape(obs_shape, action_shape, goal_shape):
    input_shape = [obs_shape]
    if action_shape is not None:
        input_shape.append(action_shape)
    if goal_shape is not None:
        input_shape.append(goal_shape)
    if len(input_shape) == 1:
        return input_shape[0]
    else:
        return tuple(input_shape)


class GeneralValueFunction(ConfigurableModule):
    """Base class for a single value/Q function.
    Maps an observation, and optionally an action and/or goal, to a value.
    """

    class Config(ConfigurableModule.Config):
        mlp = MLP.Config()

    def __init__(self, cfg, obs_shape, action_shape=None, goal_shape=None):
        super().__init__(cfg)
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.goal_shape = goal_shape

        self._input_shape = _get_input_shape(obs_shape, action_shape, goal_shape)
        output_shape = torch.Size([])   # scalar
        self.net = MLP(cfg.mlp, self._input_shape, output_shape)

    @property
    def is_action_conditioned(self):
        return self.action_shape is not None

    @property
    def is_goal_conditioned(self):
        return self.goal_shape is not None

    def get_output_shape(self, input_shape):
        self.verify_input_shape(input_shape)
        return self.net.get_output_shape(input_shape)

    def forward(self, state, **kwargs):
        return self.net(state)


class GeneralValueFunctionEnsemble(ConfigurableModule):
    """Base class for value/Q function ensembles.
    Maps an observation, and optionally an action and/or goal, to a value.
    """

    shape_relevant_kwarg_keys = {'which'}

    class Config(ConfigurableModule.Config):
        ensemble = MLPEnsemble.Config(num_models=2)

    def __init__(self, cfg, obs_shape, action_shape=None, goal_shape=None):
        super().__init__(cfg)
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.goal_shape = goal_shape

        self._input_shape = _get_input_shape(obs_shape, action_shape, goal_shape)
        output_shape = torch.Size([])   # scalar
        self.ensemble = MLPEnsemble(cfg.ensemble, self._input_shape, output_shape)
        self.num_models = cfg.ensemble.num_models

    @property
    def is_action_conditioned(self):
        return self.action_shape is not None

    @property
    def is_goal_conditioned(self):
        return self.goal_shape is not None

    def _process_which(self, which, **kwargs):
        if which == 'random':
            num_models = 1
            model_indices = [random.randrange(self.num_models)]
        elif which.startswith('redq'):
            num_models = int(which[4:])
            model_indices = random.sample(range(self.num_models), num_models)
        else:
            # all will be used
            num_models = self.num_models
            model_indices = None
        new_kwargs = dict(kwargs, num_models=num_models, model_indices=model_indices)
        return model_indices, new_kwargs

    def get_output_shape(self, input_shape, **kwargs):
        self.verify_input_shape(input_shape)
        which = kwargs.pop('which')
        model_indices, new_kwargs = self._process_which(which)
        num_models = self.num_models if model_indices is None else len(model_indices)
        output_shape = self.ensemble.get_output_shape(input_shape, **new_kwargs)
        assert output_shape == torch.Size([num_models])
        if which in {'mean', 'min', 'random'} or which.startswith('redq'):
            # Leave out dimension that indexes into ensemble's models
            return output_shape[1:]
        elif which == 'all':
            return output_shape
        else:
            raise ValueError(f'Invalid ensemble option: {which}')

    def forward(self, input, **kwargs):
        """
        The `which` argument should be one of the following:
          * "random" to randomly pick a model from the ensemble to use
            for this forward pass
          * "all" to get the outputs of all models in the ensemble
          * "min" to compute the minimum of all models in the ensemble
          * "mean" to compute the average of all models in the ensemble
        """
        # Select which model(s) to use
        which = kwargs.pop('which')
        model_indices, new_kwargs = self._process_which(which, **kwargs)

        # Batched forward passes
        out = self.ensemble(input, **new_kwargs)

        # Reduction
        if which == 'random':
            out = out[:, 0]
        elif which == 'min' or which.startswith('redq'):
            out = out.min(1).values
        elif which == 'mean':
            out = out.mean(1)
        elif which == 'all':
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
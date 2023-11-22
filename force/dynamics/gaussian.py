import random

from frozendict import frozendict
import torch
import torch.distributions as D

from force.config import BaseConfig
from force.dynamics.base import DynamicsModel
from force.env.util import space_dim
from force.distributions import DiagonalGaussian
from force.nn import ConfigurableModule, MLPEnsemble, Optimizer


class GaussianDynamicsEnsemble(ConfigurableModule, DynamicsModel):
    shape_relevant_kwarg_keys = {'num_models'}

    class Config(BaseConfig):
        ensemble = MLPEnsemble.Config()
        min_std = 1e-8
        max_std = 10.0

    def __init__(self, cfg, obs_space, act_space):
        assert 0 < cfg.min_std < cfg.max_std
        ConfigurableModule.__init__(self, cfg)

        # Determine dimensions
        self.obs_dim = obs_dim = space_dim(obs_space)
        self.act_dim = act_dim = space_dim(act_space)
        in_shape = (torch.Size([obs_dim]), torch.Size([act_dim]))
        # 2x because we predict both mean and (log)std for obs and reward
        # then +1 for terminal
        out_shape = torch.Size([2 * (obs_dim + 1) + 1])

        # Create ensemble of models
        self.ensemble = MLPEnsemble(cfg.ensemble, in_shape, out_shape)
        self.num_models = cfg.ensemble.num_models

    def get_output_shape(self, input_shape, **kwargs):
        num_models = kwargs['num_models']
        assert num_models <= self.num_models
        assert input_shape == (torch.Size([num_models, self.obs_dim]),
                               torch.Size([num_models, self.act_dim]))
        return frozendict(
            next_obs_mean=torch.Size([num_models, self.obs_dim]),
            next_obs_std=torch.Size([num_models, self.obs_dim]),
            reward_mean=torch.Size([num_models]),
            reward_std=torch.Size([num_models]),
            terminal_logits=torch.Size([num_models])
        )

    def forward(self, inputs, **kwargs):
        states, actions = inputs
        outputs = self.ensemble(inputs, **kwargs)

        # Divide up the outputs
        deltas, logstds = torch.chunk(outputs[:,:,:-1], 2, dim=-1)
        terminal_logits = outputs[:,:,-1]

        # We predict the change in state, so must be added
        zeros = torch.zeros(*states.shape[:-1], 1, dtype=states.dtype, device=states.device)
        means = torch.cat((states, zeros), dim=-1) + deltas

        stds = torch.exp(logstds)
        stds = stds.clamp(self.cfg.min_std, self.cfg.max_std)

        return {
            'next_obs_mean': means[:,:,:-1],
            'next_obs_std': stds[:,:,:-1],
            'reward_mean': means[:,:,-1],
            'reward_std': stds[:,:,-1],
            'terminal_logits': terminal_logits
        }

    @staticmethod
    def distribution_for_outputs(out):
        return {
            'next_obs': DiagonalGaussian(out['next_obs_mean'], out['next_obs_std']),
            'reward': D.Normal(out['reward_mean'], out['reward_std']),
            'terminal': D.Bernoulli(logits=out['terminal_logits'])
        }

    def distribution(self, states, actions, model_indices=None):
        if model_indices is None:
            model_indices = torch.arange(self.num_models)
        num_models = len(model_indices)
        outputs = self([states, actions], num_models=num_models, model_indices=model_indices)
        return GaussianDynamicsEnsemble.distribution_for_outputs(outputs)

    def sample(self, states, actions):
        states = states.unsqueeze(1)
        actions = actions.unsqueeze(1)
        model_index = random.randrange(self.num_models)
        distributions = self.distribution(states, actions, model_indices=[model_index])
        sample = [d.sample()[:,0] for d in distributions.values()]
        sample[-1] = sample[-1].bool() # cast
        return tuple(sample)

    def mean(self, states, actions):
        states = states.unsqueeze(1).expand(-1, self.num_models, -1)
        actions = actions.unsqueeze(1).expand(-1, self.num_models, -1)
        distributions = self.distribution(states, actions)
        return (
            distributions['next_obs'].loc.mean(1),
            distributions['reward'].loc.mean(1),
            distributions['terminal'].probs.mean(1)
        )
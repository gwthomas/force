import random

from frozendict import frozendict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from force import defaults
from force.config import BaseConfig
from force.env.util import space_dim
from force.distributions import DiagonalGaussian
from force.nn import ConfigurableModule, MLPEnsemble, Optimizer
from force.nn.util import get_device


class GaussianDynamicsEnsemble(ConfigurableModule):
    shape_relevant_kwarg_keys = {'num_models'}

    class Config(BaseConfig):
        ensemble = MLPEnsemble
        optimizer = Optimizer
        batch_size = defaults.BATCH_SIZE
        std_bound_loss_weight = 0.01

    def __init__(self, cfg, obs_space, act_space, device=None):
        ConfigurableModule.__init__(self, cfg)
        device = get_device(device)

        # Determine dimensions
        self.obs_dim = obs_dim = space_dim(obs_space)
        self.act_dim = act_dim = space_dim(act_space)
        in_shape = (torch.Size([obs_dim]), torch.Size([act_dim]))
        # 2x because we predict both mean and (log)std for obs and reward
        # then +1 for terminal
        out_shape = torch.Size([2 * (obs_dim + 1) + 1])

        # Create ensemble of models
        self.ensemble = MLPEnsemble(cfg.ensemble, in_shape, out_shape).to(device)
        self.num_models = cfg.ensemble.num_models

        # Variables for min/max logstd
        self.min_logstd = nn.Parameter(-5*torch.ones(self.obs_dim + 1, device=device))
        self.max_logstd = nn.Parameter(torch.zeros(self.obs_dim + 1, device=device))

        # Optimizer
        parameters = [*self.ensemble.parameters(), self.min_logstd, self.max_logstd]
        self.optimizer = Optimizer(cfg.optimizer, parameters)

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

        logstds = self.max_logstd - F.softplus(self.max_logstd - logstds)
        logstds = self.min_logstd + F.softplus(logstds - self.min_logstd)
        stds = torch.exp(logstds)

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

    def update(self, buffer):
        # Sample and reshape batch
        batch = buffer.sample(self.num_models * self.cfg.batch_size)

        # Special handling for ensembles
        if self.num_models > 1:
            batch_dims = (self.cfg.batch_size, self.num_models)
            for k in batch.keys():
                v = batch[k]
                if v.ndim > 1:  # vectors
                    batch[k] = v.reshape(*batch_dims, -1)
                else:           # scalars
                    batch[k] = v.reshape(*batch_dims)

        # Model forward
        distributions = self.distribution(batch['observations'], batch['actions'])

        # Optimize NLL loss
        total_log_prob = distributions['next_obs'].log_prob(batch['next_observations']) + \
                         distributions['reward'].log_prob(batch['rewards']) + \
                         distributions['terminal'].log_prob(batch['terminals'].float())
        nll_loss = -total_log_prob.mean()
        std_bound_loss = torch.sum(self.max_logstd) - torch.sum(self.min_logstd)
        loss = nll_loss + self.cfg.std_bound_loss_weight * std_bound_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
import random

import torch
from torch import nn, Size
import torch.nn.functional as F
import torch.distributions as D

from force import defaults
from force.config import BaseConfig
from force.distributions import diagonal_gaussian
from force.nn import ConfigurableModule, Optimizer
from force.nn.models import MLP
from force.nn.normalization import Normalizer
from force.nn.util import get_device, torchify, batch_iterator, freepeat, keywise_stack


class GaussianDynamicsEnsemble(ConfigurableModule):
    class Config(BaseConfig):
        net = MLP.Config
        num_models = int
        optimizer = Optimizer.Config
        batch_size = defaults.BATCH_SIZE
        std_bound_loss_weight = 0.01

    def __init__(self, cfg, env_info,
                 device=None, termination_fn=None):
        ConfigurableModule.__init__(self, cfg, device=device)

        self.termination_fn = termination_fn

        # Determine dimensions
        obs_shape = env_info.observation_shape
        act_shape = env_info.action_shape
        self.state_dim = obs_shape.numel()
        self.action_dim = act_shape.numel()
        # 2x because we predict both mean and (log)std for obs and reward
        out_shape = torch.Size([2 * (self.state_dim + 1)])

        # State will be normalized before passing to model
        self.normalizer = Normalizer(state_shape, device=self.device)

        # Create ensemble of models
        self.num_models = cfg.num_models
        self.models = nn.ModuleList([
            MLP(cfg.net, (state_shape, act_shape), out_shape)
            for _ in range(self.num_models)
        ])

        # Variables for min/max logstd
        self.min_logstd = nn.Parameter(-10*torch.ones(self.state_dim + 1, device=self.device))
        self.max_logstd = nn.Parameter(torch.zeros(self.state_dim + 1, device=self.device))

        # Optimizer
        parameters = [
            *self.models.parameters(),
            self.min_logstd, self.max_logstd
        ]
        self.optimizer = Optimizer(cfg.optimizer, parameters)

    def get_output_shape(self, input_shape, **kwargs):
        num_models = kwargs['num_models']
        assert num_models <= self.cfg.num_models
        assert input_shape == (Size([num_models, self.state_dim]),
                               Size([num_models, self.action_dim]))
        return {
            'next_state_mean': Size([num_models, self.state_dim]),
            'next_state_std': Size([num_models, self.state_dim]),
            'reward_mean': Size([num_models]),
            'reward_std': Size([num_models])
        }

    def forward(self, inputs, **kwargs):
        states, actions = inputs
        normalized_states = self.normalizer(states)

        outputs = torch.stack([
            self.models[model_idx]([normalized_states[:,data_idx,:], actions[:,data_idx,:]])
            for data_idx, model_idx in enumerate(kwargs['model_indices'])
        ], dim=1)

        # Divide up the outputs
        means, logstds = torch.chunk(outputs, 2, dim=-1)

        # logstds = self.max_logstd - F.softplus(self.max_logstd - logstds)
        # logstds = self.min_logstd + F.softplus(logstds - self.min_logstd)
        logstds = torch.clamp(logstds, min=self.min_logstd, max=self.max_logstd)
        stds = torch.exp(logstds)

        return {
            'next_state_mean': states + means[:,:,:-1], # we predict the state delta
            'next_state_std': stds[:,:,:-1],
            'reward_mean': means[:,:,-1],
            'reward_std': stds[:,:,-1]
        }

    @staticmethod
    def distribution_for_outputs(out):
        return (
            diagonal_gaussian(out['next_state_mean'], out['next_state_std']),
            D.Normal(out['reward_mean'], out['reward_std'])
        )

    def distribution(self, states, actions, model_indices=None):
        if model_indices is None:
            model_indices = torch.arange(self.cfg.num_models)
        num_models = len(model_indices)
        outputs = self(
            [states, actions],
            num_models=num_models, model_indices=model_indices
        )
        return GaussianDynamicsEnsemble.distribution_for_outputs(outputs)

    def sample(self, states, actions, model_index=None):
        if model_index is None:
            model_index = random.randrange(self.cfg.num_models)
        else:
            assert isinstance(model_index, int)
            assert 0 <= model_index < self.cfg.num_models
        states = states.unsqueeze(1)
        actions = actions.unsqueeze(1)
        model_indices = [model_index]
        next_state_distr, reward_distr = self.distribution(
            states, actions, model_indices=model_indices
        )
        next_state = next_state_distr.sample()[:,0]
        reward = reward_distr.sample()[:,0]
        terminal = self.termination_fn(next_state)
        return next_state, reward, terminal

    def mean(self, states, actions):
        states = freepeat(states, 1, dim=1)
        actions = freepeat(actions, 1, dim=1)
        next_state_distr, reward_distr = self.distribution(states, actions)
        mean_next_state = next_state_distr.loc.mean(1)
        mean_reward = reward_distr.loc.mean(1)
        terminal = self.termination_fn(mean_next_state)
        return mean_next_state, mean_reward, terminal

    def log_likelihood(self, batch: dict):
        next_state_distr, reward_distr = self.distribution(batch['observations'], batch['actions'])
        return next_state_distr.log_prob(batch['next_observations']) + reward_distr.log_prob(batch['rewards'])

    def update(self, batch: dict):
        # Negative log-likelihood loss
        nll_loss = -self.log_likelihood(batch).mean()

        # Loss to make logstd bounds tighter
        std_bound_loss = torch.sum(self.max_logstd) - torch.sum(self.min_logstd) + \
                         torch.sum(F.relu(self.min_logstd - self.max_logstd))

        # Total loss
        loss = nll_loss + self.cfg.std_bound_loss_weight * std_bound_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def epoch(self, data: dict):
        # Maintain a separate minibatch ordering for each model
        iterators = [
            batch_iterator(data, self.cfg.batch_size, shuffle=True)
            for _ in range(self.num_models)
        ]

        while True:
            try:
                batch = torchify(
                    keywise_stack([next(it) for it in iterators], dim=1),
                    device=self.device
                )
            except StopIteration:
                break

            loss = self.update(batch)
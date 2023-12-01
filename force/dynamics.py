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
from force.nn.util import get_device, freepeat
from force.util import batch_iterator


class GaussianDynamicsEnsemble(ConfigurableModule):
    shape_relevant_kwarg_keys = {'num_models'}

    class Config(BaseConfig):
        ensemble = MLPEnsemble
        terminal_discriminator = MLPEnsemble
        optimizer = Optimizer
        batch_size = defaults.BATCH_SIZE
        std_bound_loss_weight = 0.01

    def __init__(self, cfg, obs_space, act_space,
                 device=None, termination_fn=None):
        ConfigurableModule.__init__(self, cfg)
        device = get_device(device)

        self.termination_fn = termination_fn

        # Determine dimensions
        self.obs_dim = obs_dim = space_dim(obs_space)
        self.act_dim = act_dim = space_dim(act_space)
        obs_shape = torch.Size([obs_dim])
        act_shape = torch.Size([act_dim])
        # 2x because we predict both mean and (log)std for obs and reward
        out_shape = torch.Size([2 * (obs_dim + 1)])

        # Create ensemble of models
        self.ensemble = MLPEnsemble(cfg.ensemble, (obs_shape, act_shape), out_shape).to(device)
        self.num_models = cfg.ensemble.num_models

        # Variables for min/max logstd
        self.min_logstd = nn.Parameter(-10*torch.ones(self.obs_dim + 1, device=device))
        self.max_logstd = nn.Parameter(torch.zeros(self.obs_dim + 1, device=device))

        # Optimizer
        parameters = [
            *self.ensemble.parameters(),
            self.min_logstd, self.max_logstd
        ]

        if termination_fn is None:
            self.terminal_discriminator = MLPEnsemble(cfg.terminal_discriminator, obs_shape, torch.Size([])).to(device)
            assert cfg.ensemble.num_models == cfg.terminal_discriminator.num_models
            parameters.extend(self.terminal_discriminator.parameters())
        else:
            self.terminal_discriminator = None

        self.optimizer = Optimizer(cfg.optimizer, parameters)

    def get_output_shape(self, input_shape, **kwargs):
        num_models = kwargs['num_models']
        assert num_models <= self.num_models
        assert input_shape == (torch.Size([num_models, self.obs_dim]),
                               torch.Size([num_models, self.act_dim]))
        return frozendict(
            next_state_mean=torch.Size([num_models, self.obs_dim]),
            next_state_std=torch.Size([num_models, self.obs_dim]),
            reward_mean=torch.Size([num_models]),
            reward_std=torch.Size([num_models])
        )

    def forward(self, inputs, **kwargs):
        states, actions = inputs
        outputs = self.ensemble(inputs, **kwargs)

        # Divide up the outputs
        means, logstds = torch.chunk(outputs, 2, dim=-1)

        logstds = self.max_logstd - F.softplus(self.max_logstd - logstds)
        logstds = self.min_logstd + F.softplus(logstds - self.min_logstd)
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
            DiagonalGaussian(out['next_state_mean'], out['next_state_std']),
            D.Normal(out['reward_mean'], out['reward_std'])
        )

    def distribution(self, states, actions, model_indices=None):
        if model_indices is None:
            model_indices = torch.arange(self.num_models)
        num_models = len(model_indices)
        outputs = self(
            [states, actions],
            num_models=num_models, model_indices=model_indices
        )
        return GaussianDynamicsEnsemble.distribution_for_outputs(outputs)

    def terminal_distribution(self, next_states, model_indices=None):
        assert next_states.ndim == 3
        assert next_states.shape[-1] == self.obs_dim
        if self.termination_fn is None:
            if model_indices is None:
                model_indices = torch.arange(self.num_models)
            num_models = len(model_indices)
            outputs = self.terminal_discriminator(
                next_states,
                num_models=num_models, model_indices=model_indices
            )
            return D.Bernoulli(logits=outputs)
        else:
            lead_dims = next_states.shape[:2]
            batched_next_states = next_states.reshape(-1, self.obs_dim)
            batched_terminals = self.termination_fn(batched_next_states)
            assert batched_terminals.shape == (lead_dims[0] * lead_dims[1],)
            terminals = batched_terminals.reshape(*lead_dims)
            return D.Bernoulli(probs=terminals.float())

    def sample(self, states, actions, model_index=None):
        if model_index is None:
            model_index = random.randrange(self.num_models)
        else:
            assert isinstance(model_index, int)
            assert 0 <= model_index < self.num_models
        states = states.unsqueeze(1)
        actions = actions.unsqueeze(1)
        model_indices = [model_index]
        next_state_distr, reward_distr = self.distribution(
            states, actions, model_indices=model_indices
        )
        next_state = next_state_distr.sample()
        reward = reward_distr.sample()
        terminal_distr = self.terminal_distribution(
            next_state, model_indices=model_indices
        )
        terminal = terminal_distr.sample().bool()
        return next_state[:,0], reward[:,0], terminal[:,0]

    def mean(self, states, actions):
        states = freepeat(states, 1, dim=1)
        actions = freepeat(actions, 1, dim=1)
        next_state_distr, reward_distr = self.distribution(states, actions)
        mean_next_state = next_state_distr.loc.mean(1)
        mean_reward = reward_distr.loc.mean(1)
        if self.termination_fn is None:
            mean_terminal_prob = F.sigmoid(self.terminal_discriminator(mean_next_state)).mean(1)
        else:
            mean_terminal_prob = self.termination_fn(mean_next_state)
        return mean_next_state, mean_reward, mean_terminal_prob

    def log_likelihood(self, states, actions, next_states, rewards, terminals):
        next_state_distr, reward_distr = self.distribution(states, actions)
        ll = next_state_distr.log_prob(next_states) + reward_distr.log_prob(rewards)
        if self.termination_fn is None:
            terminal_distr = self.terminal_distribution(next_states)
            ll = ll + terminal_distr.log_prob(terminals.float())
        return ll

    def update(self, states, actions, next_states, rewards, terminals):
        # Negative log-likelihood loss
        nll_loss = -self.log_likelihood(states, actions, next_states, rewards, terminals).mean()

        # Loss to make logstd bounds tighter
        std_bound_loss = torch.sum(self.max_logstd) - torch.sum(self.min_logstd)

        # Total loss
        loss = nll_loss + self.cfg.std_bound_loss_weight * std_bound_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def epoch(self, states, actions, next_states, rewards, terminals):
        # For simplicity, this method *does not* guarantee a full, unique
        # pass over the dataset for every model.
        # We want the models to look at different batches to reduce correlation.
        # Our strategy is to make num_models passes over the dataset, where
        # each model sees only 1/num_models fraction of the dataset each time.
        tensors = [states, actions, next_states, rewards, terminals]
        ensemble_batch_size = self.num_models * self.cfg.batch_size

        for _ in range(self.num_models):
            for batch in batch_iterator(tensors, ensemble_batch_size, shuffle=True):
                # Preprocess batch
                for i in range(len(tensors)):
                    x = batch[i]

                    # Handle truncation
                    leftover = len(x) % self.num_models
                    if leftover != 0:
                        x = x[:-leftover]

                    # Reshape
                    batch_dims = (len(x) // self.num_models, self.num_models)
                    if x.ndim > 1:
                        batch[i] = x.reshape(*batch_dims, -1)
                    else:
                        batch[i] = x.reshape(*batch_dims)

                # If leftover < num_models, batch is empty, so skip it
                if len(batch[0]) == 0:
                    continue

                self.update(*batch)
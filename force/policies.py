from abc import ABC, abstractmethod
import math

from frozendict import frozendict
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Space
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

from force.distributions import diagonal_gaussian
from force.nn import ConfigurableModule
from force.nn.models.mlp import MLP
from force.nn.util import get_device, torchify, freepeat


class Policy(ABC):
    @abstractmethod
    def act(self, obs, eval):
        raise NotImplementedError

    def act1(self, obs, eval):
        with torch.no_grad():
            return self.act(torch.unsqueeze(obs, 0), eval)[0]


class UniformPolicy(Policy):
    def __init__(self, env_or_action_space, device=None):
        if isinstance(env_or_action_space, gym.Env):
            action_space = env_or_action_space.action_space
        elif isinstance(env_or_action_space, gym.Space):
            action_space = env_or_action_space
        else:
            raise ValueError('Must pass env or action space')

        if isinstance(action_space, Box):
            self.discrete = False
            self.low = torchify(action_space.low, device=device)
            self.high = torchify(action_space.high, device=device)
            self.shape = tuple(action_space.shape)
        elif isinstance(action_space, Discrete):
            self.discrete = True
            self.n = action_space.n
        else:
            raise NotImplementedError(f'Unsupported action space: {action_space}')

        self.device = get_device(device)

    def act(self, obs, eval):
        batch_size = len(obs)
        if self.discrete:
            return torch.randint(self.n, size=(batch_size,), device=self.device)
        else:
            return self.low + torch.rand(batch_size, *self.shape, device=self.device) * (self.high - self.low)

    def prob(self, actions):
        batch_size = len(actions)
        if self.discrete:
            assert actions.dim() == 1
            p = 1./self.n
        else:
            assert actions.dim() == 2
            p = 1./torch.prod(self.high - self.low)
        return torch.full([batch_size], p, device=self.device)

    def log_prob(self, actions):
        return torch.log(self.prob(actions))


class GaussianNoiseWrapper(Policy):
    def __init__(self, policy, std, noise_clip=0.5, action_clip=1):
        self.policy = policy
        self.std = std
        self.noise_clip = noise_clip
        self.action_clip = action_clip

    def act(self, states, eval):
        noiseless_actions = self.policy.act(states, eval)
        if eval:
            return noiseless_actions
        else:
            noise = torch.clamp(
                self.std * torch.randn_like(noiseless_actions),
                min=-self.noise_clip, max=self.noise_clip
            )
            return torch.clamp(
                noiseless_actions + noise,
                min=-self.action_clip, max=self.action_clip
            )


class MixturePolicy(Policy):
    def __init__(self, policies, probs=None):
        if probs is not None:
            assert sum(probs) == 1.
            assert len(policies) == len(probs)
        self.policies = policies
        self.probs = probs

    def act(self, states, eval):
        index = np.random.choice(len(self.policies), p=self.probs)
        return self.policies[index].act(states, eval)


class NeuralPolicy(Policy, ConfigurableModule):
    class Config(ConfigurableModule.Config):
        net = MLP.Config

    def __init__(self, cfg, input_shape, output_shape, final_activation=None):
        ConfigurableModule.__init__(self, cfg)
        self.net = MLP(cfg.net, input_shape, output_shape,
                       final_activation=final_activation)


class DeterministicNeuralPolicy(NeuralPolicy):
    def __init__(self, cfg, input_shape, action_shape, squash=True):
        super().__init__(cfg, input_shape, action_shape,
                         final_activation=('tanh' if squash else None))

    def act(self, obs, eval):
        return self.net(obs).clamp(-1, 1)


class DistributionalNeuralPolicy(NeuralPolicy):
    use_special_eval = False

    @abstractmethod
    def _distribution(self, net_out):
        """Construct a distribution from the output(s) of the network"""
        raise NotImplementedError

    def distribution(self, obs):
        return self._distribution(self.net(obs))

    def _special_eval(self, distr):
        """Optionally override to implement e.g. deterministic eval.
        Must set use_special_eval = True or this will not be called."""
        raise NotImplementedError

    def act(self, obs, eval: bool):
        distr = self.distribution(obs)
        if eval and self.use_special_eval:
            return self._special_eval(distr)
        else:
            return distr.sample()


class BoltzmannPolicy(DistributionalNeuralPolicy):
    use_special_eval = True

    def __init__(self, cfg, input_shape, output_shape):
        super().__init__(cfg, input_shape, output_shape)
        self.temperature = 1

    def _distribution(self, net_out):
        return D.Categorical(logits=net_out/self.temperature)

    def _special_eval(self, distr):
        return distr.logits.argmax(dim=-1)


class GaussianPolicy(DistributionalNeuralPolicy):
    class Config(DistributionalNeuralPolicy.Config):
        use_state_dependent_std = False
        init_std = 0.5
        min_logstd = -20.0
        max_logstd = 2.0
        squash = False

    use_special_eval = True

    def __init__(self, cfg, input_shape, action_shape):
        if cfg.use_state_dependent_std:
            output_shape = frozendict({'mean': action_shape, 'logstd': action_shape})
        else:
            output_shape = action_shape
        super().__init__(cfg, input_shape, output_shape)

        # Initialize log std
        init_log_std = math.log(cfg.init_std)
        if not cfg.use_state_dependent_std:
            self.logstd = nn.Parameter(torch.full(action_shape, init_log_std))
        else:
            linear_layer = self.net[-2]
            assert isinstance(linear_layer, nn.Linear)
            linear_layer.bias.data[:] = init_log_std

        self.last_pretanh = None

    def act(self, states, eval):
        actions = super().act(states, eval)
        if self.cfg.squash:
            # tanh ensures samples already in [-1,1]
            return actions
        else:
            # The mean lies within [-1,1], but samples may have gone beyond the
            # bounds, necessitating clipping.
            return actions.clamp(-1, 1)

    def _distribution(self, net_out):
        cfg = self.cfg
        if cfg.use_state_dependent_std:
            mean = net_out['mean']
            logstd = net_out['logstd']
            logstd = logstd.clamp(min=cfg.min_logstd, max=cfg.max_logstd)
            std = torch.exp(logstd)
        else:
            mean = net_out
            logstd = self.logstd
            logstd.data.clamp_(min=cfg.min_logstd, max=cfg.max_logstd)
            std = freepeat(torch.exp(logstd), mean.size(0), dim=0)

        if cfg.squash:
            self.last_pretanh = mean
            mean = torch.tanh(mean)
        return diagonal_gaussian(mean, std, squash=cfg.squash)

    def _special_eval(self, distr):
        return distr.mode


class NormalizedTanhPolicy(NeuralPolicy):
    class Config(NeuralPolicy.Config):
        noise_std = 0.29

    def act(self, obs, eval):
        means = self.net(obs)

        # Normalize
        G = means.abs().mean(1, keepdim=True)
        means = means / G.clamp(min=1)

        # Optionally add Gaussian noise
        if eval:
            pretanh = means
        else:
            pretanh = means + self.cfg.noise_std * torch.randn_like(means)

        return torch.tanh(pretanh)
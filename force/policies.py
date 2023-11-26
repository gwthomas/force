from abc import ABC, abstractmethod

import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Space
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

from force.distributions import DiagonalGaussian, SquashedGaussian
from force.nn import ConfigurableModule, MLP
from force.nn.util import get_device, torchify


class BasePolicy(ABC):
    @abstractmethod
    def act(self, states, eval):
        raise NotImplementedError

    def act1(self, state, eval):
        with torch.no_grad():
            return self.act(torch.unsqueeze(state, 0), eval)[0]


class UniformPolicy(BasePolicy):
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

    def act(self, states, eval):
        batch_size = len(states)
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


class GaussianNoiseWrapper(BasePolicy):
    def __init__(self, policy, std, noise_clip=0.5, action_clip=1.0):
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


class MixturePolicy(BasePolicy):
    def __init__(self, policies, probs=None):
        if probs is not None:
            assert sum(probs) == 1.
            assert len(policies) == len(probs)
        self.policies = policies
        self.probs = probs

    def act(self, states, eval):
        index = np.random.choice(len(self.policies), p=self.probs)
        return self.policies[index].act(states, eval)


class DeterministicPolicy(BasePolicy, ConfigurableModule):
    class Config(ConfigurableModule.Config):
        mlp = MLP

    def __init__(self, cfg, input_shape, output_shape, squash=True):
        ConfigurableModule.__init__(self, cfg)
        self.net = MLP(cfg.mlp, input_shape, output_shape,
                       final_activation=('tanh' if squash else None))

    def act(self, states, eval):
        return self.net(states)


class StochasticPolicy(BasePolicy, ConfigurableModule):
    use_special_eval = False

    class Config(ConfigurableModule.Config):
        mlp = MLP

    def __init__(self, cfg, input_shape, output_shape):
        ConfigurableModule.__init__(self, cfg)
        self.net = MLP(cfg.mlp, input_shape, output_shape)

    @abstractmethod
    def _distr(self, *network_outputs):
        raise NotImplementedError

    def distr(self, states):
        return self._distr(self.net(states))

    @abstractmethod
    def _special_eval(self, distr):
        raise NotImplementedError

    def act(self, states, eval):
        distr = self.distr(states)
        if eval and self.use_special_eval:
            return self._special_eval(distr)
        else:
            return distr.sample()


class BoltzmannPolicy(StochasticPolicy):
    use_special_eval = True

    def _distr(self, logits):
        return D.Categorical(logits=logits)

    def _special_eval(self, distr):
        return distr.logits.argmax(dim=-1)


class GaussianPolicy(StochasticPolicy):
    class Config(StochasticPolicy.Config):
        init_std = 0.1

    use_special_eval = True

    def __init__(self, cfg, input_shape, action_dim):
        output_shape = torch.Size([action_dim])
        super().__init__(cfg, input_shape, output_shape)
        self.log_std = nn.Parameter(torch.full([action_dim], np.log(cfg.init_std)))

    def act(self, states, eval):
        actions = super().act(states, eval)
        return torch.clamp(actions, -1, 1)

    def _distr(self, net_out):
        batch_size = net_out.shape[0]
        std = self.log_std.exp().unsqueeze(0).expand(batch_size, -1)
        return DiagonalGaussian(net_out, std)

    def _special_eval(self, distr):
        return distr.loc


class SquashedGaussianPolicy(StochasticPolicy):
    class Config(StochasticPolicy.Config):
        logstd_min = -20.0
        logstd_max = 2.0

    use_special_eval = True

    def __init__(self, cfg, input_shape, action_dim):
        output_shape = torch.Size([action_dim*2])
        super().__init__(cfg, input_shape, output_shape)

    def _distr(self, net_out):
        mu, logstd = net_out.chunk(2, dim=-1)

        # constrain log_std inside (log_std_min, log_std_max)
        cfg = self.cfg
        logstd = cfg.logstd_min + (cfg.logstd_max - cfg.logstd_min) * torch.sigmoid(logstd)
        return SquashedGaussian(mu, logstd.exp())

    def _special_eval(self, distr):
        return distr.mean
from abc import ABC, abstractmethod

from gym.spaces import Box, Discrete
import numpy as np
import torch
import torch.nn as nn
from torch import distributions as td

from force.torch_util import torchify, Module, device
from force.etc.squashed_gaussian import SquashedGaussian


class BasePolicy(ABC):
    @abstractmethod
    def act(self, states, eval): pass

    def act1(self, state, eval=False):
        return self.act(torch.unsqueeze(state, 0), eval)[0]


class RandomPolicy(BasePolicy):
    def __init__(self, action_space):
        assert isinstance(action_space, Box) or isinstance(action_space, Discrete), \
               f'Unimplemented space type {action_space}'
        self.action_space = action_space

    def act(self, states, eval):
        batch_size = len(states)
        if isinstance(self.action_space, Box):
            low = torchify(self.action_space.low, to_device=True)
            high = torchify(self.action_space.high, to_device=True)
            return low + torch.rand(batch_size, *self.action_space.shape, device=device) * (high - low)
        elif isinstance(self.action_space, Discrete):
            return torch.randint(self.action_space.n, size=(batch_size,), device=device)
        else:
            raise NotImplementedError(f'Unimplemented space type {self.action_space}')


class NoisyPolicy(BasePolicy):
    def __init__(self, policy, noise, max_action=1):
        self.policy = policy
        self.noise = noise
        self.max_action = max_action
        self.std = noise * max_action

    def act(self, states, eval):
        noiseless_actions = self.policy.act(states, eval)
        noisy_actions = noiseless_actions + torch.normal(0, self.std, size=noiseless_actions.shape).to(device)
        return noisy_actions.clamp(-self.max_action, self.max_action)


class EpsilonRandomPolicy(BasePolicy):
    def __init__(self, policy, action_space, epsilon):
        self.policy = policy
        self.random_policy = RandomPolicy(action_space)
        self.epsilon = epsilon

    def act(self, states, eval):
        actions = self.policy.act(states, eval)
        random_actions = self.random_policy.act(states, eval)
        mask = np.random.binomial(1, self.epsilon, size=len(states))
        mask = np.expand_dims(mask, 1)
        return mask * random_actions + (1-mask) * actions


class TorchPolicy(BasePolicy, Module):
    def __init__(self, net):
        Module.__init__(self)
        self.net = net
        self.use_special_eval = False

    @abstractmethod
    def _distr(self, *network_outputs): pass

    def distr(self, states):
        return self._distr(self.net(states))

    @abstractmethod
    def _special_eval(self, distr):
        raise NotImplementedError

    def act(self, states, eval):
        with torch.no_grad():
            distr = self.distr(states)

        if self.use_special_eval:
            action = self._special_eval(distr)
        else:
            action = distr.sample()

        return action


class BoltzmannPolicy(TorchPolicy):
    def _distr(self, logits):
        return td.Categorical(logits=logits)

    def _special_eval(self, distr):
        # TODO
        raise NotImplementedError


class GaussianPolicy(TorchPolicy):
    def __init__(self, net, action_dim, init_std=0.1):
        super().__init__(net)
        self.log_std = nn.Parameter(torch.full([action_dim], np.log(init_std)))

    def act(self, states, eval):
        actions = super().act(states, eval)
        return torch.clamp(actions, -1, 1)

    def _distr(self, net_out):
        return td.Independent(td.Normal(net_out, self.log_std.exp()), 1)

    def _special_eval(self, distr):
        return distr.mean


class SquashedGaussianPolicy(TorchPolicy):
    def __init__(self, net, log_std_bounds=[-5,2]):
        super().__init__(net)
        self.log_std_bounds = log_std_bounds

    def _distr(self, net_out):
        mu, log_std = net_out.chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        return td.Independent(SquashedGaussian(mu, std), 1)

    def _special_eval(self, distr):
        return distr.mean
import numpy as np
import torch
import torch.nn as nn


from ..env.util import env_dims
from ..policy import BasePolicy
from ..torch_util import Module, mlp
from ..train import get_optimizer, epochal_training, supervised_loss, L2Loss


class BehavioralCloning(Module, BasePolicy):
    def __init__(self, env, hidden_dims=(256, 256)):
        nn.Module.__init__(self)
        state_dim, action_dim = env_dims(env)
        self.max_action = float(env.action_space.high[0])
        assert np.array_equal(env.action_space.high, self.max_action * np.ones_like(env.action_space.high))
        assert np.array_equal(env.action_space.low, -self.max_action * np.ones_like(env.action_space.high))
        self.net = mlp([state_dim, *hidden_dims, action_dim], output_activation=nn.Tanh)
        self.optimizer = get_optimizer(self.net)

    def forward(self, state):
        return self.max_action * self.net(state)

    def act(self, state, eval):
        with torch.no_grad():
            return self(state)

    def train(self, states, actions, epochs=25, **kwargs):
        compute_loss = supervised_loss(self, L2Loss())
        epochal_training(compute_loss, self.optimizer, [states, actions], epochs, **kwargs)
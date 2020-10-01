import numpy as np
import torch
import torch.nn as nn

from force.torch_util import mlp, Module, torchify, freeze_module, update_ema
from force.train import get_optimizer


class ValueFunction(Module):
    def __init__(self, state_dim, hidden_dim=256, hidden_depth=2, discount=0.99):
        super().__init__()
        self.state_dim = state_dim
        self.discount = discount
        self.net = mlp([state_dim] + ([hidden_dim] * hidden_depth) + [1])
        self.target_net = mlp([state_dim] + ([hidden_dim] * hidden_depth) + [1])
        freeze_module(self.target_net)
        self.target_net.load_state_dict(self.net.state_dict())

        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = get_optimizer(self.net)

    def __call__(self, states):
        with torch.no_grad():
            return np.squeeze(self.net(torchify(states)).cpu().numpy(), axis=-1)

    def train(self, replay_buffer, iters=1, batch_size=256, tau=0.005):
        losses = []
        for _ in range(iters):
            states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size)
            rewards = rewards.unsqueeze(1)
            not_dones = (1 - dones).unsqueeze(1)

            v_next = self.target_net(next_states)
            v_target = rewards + not_dones * self.discount * v_next
            loss = self.loss_fn(self.net(states), v_target)
            losses.append(float(loss.item()))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            update_ema(self.target_net, self.net, tau)

        return np.mean(losses)

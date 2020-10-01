import copy

import torch.nn as nn

from ..policy import BasePolicy
from ..train import get_optimizer
from ..torch_util import Module, freeze_module


class ActorCritic(Module, BasePolicy):
    def __init__(self, actor, critic,
                 use_actor_target=False, use_critic_target=False,
                 actor_optimizer=None, critic_optimizer=None):
        Module.__init__(self)
        self.actor = actor
        self.critic = critic

        if isinstance(actor, nn.Module):
            self.actor_optimizer = get_optimizer(actor, actor_optimizer)
        if isinstance(critic, nn.Module):
            self.critic_optimizer = get_optimizer(critic, critic_optimizer)

        if use_actor_target:
            self.actor_target = copy.deepcopy(actor)
            freeze_module(self.actor_target)
        else:
            self.actor_target = None

        if use_critic_target:
            self.critic_target = copy.deepcopy(critic)
            freeze_module(self.critic_target)
        else:
            self.critic_target = None

    def act(self, states, eval):
        return self.actor.act(states, eval)
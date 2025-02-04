import random

from force.alg.sac import SAC
from force.alg.sop import SOP


class REDQSAC(SAC):
    class Config(SAC.Config):
        utd_ratio = 20  # G
        num_min = 2     # M

    def compute_target_value(self, obs):
        distr = self.actor.distribution(obs)
        action = distr.sample()
        indices = random.sample(range(self.target_critic.num_models), self.cfg.M)
        value = self.target_critic([obs, action], which=tuple(indices)).min(1).values
        if not self.cfg.deterministic_backup:
            value = value - self.alpha * distr.log_prob(action)
        return value

    def _update(self, counters: dict):
        # Update critic ensemble G times
        for _ in range(self.cfg.utd_ratio):
            minibatch = self.get_minibatch()
            self.update_critic(minibatch)
            self.update_target_networks()

        # Update actor once
        self.update_actor(minibatch['observations'])


class REDQSOP(SOP):
    class Config(SOP.Config):
        utd_ratio = 20  # G
        num_min = 2     # M

    def compute_target_value(self, obs):
        noisy_actions = self.actor.act(obs, eval=False)
        indices = random.sample(range(self.target_critic.num_models), self.cfg.num_min)
        return self.target_critic([obs, noisy_actions], which=tuple(indices)).min(1).values

    def _update(self, counters: dict):
        # Update critic ensemble G times
        for _ in range(self.cfg.utd_ratio):
            minibatch = self.get_minibatch()
            self.update_critic(minibatch)
            self.update_target_networks()

        # Update actor once
        self.update_actor(minibatch['observations'])
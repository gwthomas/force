from force.alg.sac import SAC


class REDQ(SAC):
    class Config(SAC.Config):
        G = 20
        M = 2

    def compute_value(self, obs, use_target_network):
        critic = self.critic_target if use_target_network else self.critic
        distr = self.actor.distr(obs)
        action = distr.sample()
        value = critic([obs, action], which=f'redq{self.cfg.M}')
        if not self.cfg.deterministic_backup:
            value = value - self.alpha * distr.log_prob(action)
        return value

    def update(self, buffer, counters):
        # Update critic ensemble G times
        for _ in range(self.cfg.G):
            batch = buffer.sample(self.cfg.batch_size)
            self.update_critic(batch)
            self.update_target_networks()

        # Update actor once
        self.update_actor(batch)
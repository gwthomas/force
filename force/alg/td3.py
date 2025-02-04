from force.alg.actor_critic import BufferedActorCritic
from force.policies import (
    PolicyMode, UnsupportedPolicyMode,
    DeterministicNeuralPolicy, GaussianNoiseWrapper
)


class TD3(BufferedActorCritic):
    class Config(BufferedActorCritic.Config):
        actor = DeterministicNeuralPolicy.Config
        explore_noise = 0.1
        smooth_noise = 0.2
        noise_clip = 0.5
        delay = 2

    def __init__(self, cfg, env_info,
                 actor=None, device=None):
        if actor is None:
            actor = DeterministicNeuralPolicy(
                cfg.actor, env_info.observation_shape, env_info.action_shape
            )

        super().__init__(
            cfg, env_info, actor,
            use_target_actor=True, use_target_critic=True,
            device=device
        )

        # Noisy versions of the policy for exploration and smoothing
        self.acting_policy = GaussianNoiseWrapper(
            self.actor, cfg.explore_noise,
            noise_clip=cfg.noise_clip
        )
        self.smoothing_policy = GaussianNoiseWrapper(
            self.target_actor, cfg.smooth_noise,
            noise_clip=cfg.noise_clip
        )

    def act(self, obs, mode: PolicyMode):
        return self.acting_policy.act(obs, mode)

    def compute_target_value(self, obs):
        noisy_actions = self.smoothing_policy.act(obs, mode=PolicyMode.EXPLORE)
        return self.target_critic([obs, noisy_actions], which='min')

    def update_with_minibatch(self, batch: dict, counters: dict):
        self.update_critic(batch)

        # Delayed policy updates
        if counters['updates'] % self.cfg.delay == 0:
            self.update_actor(batch['observations'])
            self.update_target_networks()
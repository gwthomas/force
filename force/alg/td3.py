from force.alg.actor_critic import BufferedActorCritic
from force.config import Configurable, BaseConfig, Field
from force.env.util import space_shape
from force.policies import DeterministicNeuralPolicy, GaussianNoiseWrapper


class TD3(BufferedActorCritic):
    class Config(BufferedActorCritic.Config):
        actor = DeterministicNeuralPolicy.Config
        explore_noise = 0.1
        smooth_noise = 0.2
        noise_clip = 0.5
        delay = 2

    def __init__(self, cfg, obs_space, act_space,
                 actor=None, device=None):
        Configurable.__init__(self, cfg)
        obs_shape = space_shape(obs_space)
        act_shape = space_shape(act_space)
        if actor is None:
            actor = DeterministicNeuralPolicy(cfg.actor, obs_shape, act_shape)

        super().__init__(
            cfg, obs_space, act_space, actor,
            use_actor_target=True, use_critic_target=True,
            device=device
        )

        # Noisy versions of the policy for exploration and smoothing
        self.explore_policy = GaussianNoiseWrapper(
            self.actor, cfg.explore_noise,
            noise_clip=cfg.noise_clip
        )
        self.smooth_policy = GaussianNoiseWrapper(
            self.actor_target, cfg.smooth_noise,
            noise_clip=cfg.noise_clip
        )

    def act(self, obs, eval):
        return self.explore_policy.act(obs, eval)

    def compute_value(self, obs, use_target_network):
        critic = self.critic_target if use_target_network else self.critic
        noisy_actions = self.smooth_policy.act(obs, eval=False)
        return critic([obs, noisy_actions], which='min')

    def update_with_minibatch(self, batch: dict, counters: dict):
        self.update_critic(batch)

        # Delayed policy updates
        if counters['updates'] % self.cfg.delay == 0:
            self.update_actor(batch['observations'])
            self.update_target_networks()
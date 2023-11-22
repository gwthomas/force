from force.alg.actor_critic import ActorCritic
from force.config import Configurable, BaseConfig, Field
from force.env.util import space_shape
from force.nn.util import get_device
from force.policies import DeterministicPolicy, GaussianNoiseWrapper
from force.value_functions import QFunctionEnsemble


class TD3(ActorCritic):
    class Config(ActorCritic.Config):
        actor = DeterministicPolicy.Config()
        critic = QFunctionEnsemble.Config()
        explore_noise = 0.1
        smooth_noise = 0.2
        noise_clip = 0.5
        delay = 2

    def __init__(self, cfg, obs_space, act_space,
                 actor=None, critic=None, device=None):
        Configurable.__init__(self, cfg)
        obs_shape = space_shape(obs_space)
        act_shape = space_shape(act_space)
        if actor is None:
            actor = DeterministicPolicy(cfg.actor, obs_shape, act_shape)
        if critic is None:
            critic = QFunctionEnsemble(cfg.critic, obs_shape, act_shape)
        device = get_device(device)

        super().__init__(
            cfg, obs_space, act_space, actor, critic,
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
        smooth_actions = self.smooth_policy.act(obs, eval=False)
        return critic([obs, smooth_actions], which='min')

    def update_with_batch(self, batch: dict, counters: dict):
        self.update_critic(batch)

        # Delayed policy updates
        if counters['updates'] % self.cfg.delay == 0:
            self.update_actor(batch)
            self.update_target_networks()
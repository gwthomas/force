from force.alg.actor_critic import BufferedActorCritic
from force.config import Configurable, BaseConfig, Field
from force.policies import NormalizedTanhPolicy


class SOP(BufferedActorCritic):
    """Streamlined Off-Policy (SOP) is a basic actor-critic algorithm.
    In fact, no customization is needed relative to the BufferedActorCritic
    base class, except for a modified policy architecture that normalizes the
    outputs of the network (if they are too big) before squashing with tanh.
    """

    class Config(BufferedActorCritic.Config):
        actor = NormalizedTanhPolicy.Config

    def __init__(self, cfg, env_info,
                 actor=None, device=None):
        Configurable.__init__(self, cfg)
        if actor is None:
            actor = NormalizedTanhPolicy(
                cfg.actor, env_info.observation_shape, env_info.action_shape
            )

        super().__init__(
            cfg, env_info, actor,
            use_target_actor=False, use_target_critic=True,
            device=device
        )
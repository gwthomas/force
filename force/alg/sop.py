from force.alg.actor_critic import BufferedActorCritic
from force.config import Configurable, BaseConfig, Field
from force.env.util import space_shape
from force.policies import NormalizedTanhPolicy


class SOP(BufferedActorCritic):
    """Streamlined Off-Policy (SOP) is a basic actor-critic algorithm.
    In fact, no customization is needed relative to the BufferedActorCritic
    base class, except for a modified policy architecture that normalizes the
    outputs of the network (if they are too big) before squashing with tanh.
    """

    class Config(BufferedActorCritic.Config):
        actor = NormalizedTanhPolicy.Config

    def __init__(self, cfg, obs_space, act_space,
                 actor=None, device=None):
        Configurable.__init__(self, cfg)
        obs_shape = space_shape(obs_space)
        act_shape = space_shape(act_space)
        if actor is None:
            actor = NormalizedTanhPolicy(cfg.actor, obs_shape, act_shape)

        super().__init__(
            cfg, obs_space, act_space, actor,
            use_actor_target=False, use_critic_target=True,
            device=device
        )
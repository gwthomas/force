from .actor_critic import ActorCritic
from .base import IterativeAlgorithm, Agent, BufferedRLAlgorithm
from .iql import IQL
from .mbpo import MBPO
from .redq import REDQ
from .sac import SAC, DSAC
from .td3 import TD3


NAMED_ALGORITHMS = {
    'DSAC': DSAC,
    'IQL': IQL,
    'MBPO': MBPO,
    'REDQ': REDQ,
    'SAC': SAC,
    'TD3': TD3
}
from .iql import IQL
from .mbpo import MBPO
from .ppo import PPO
from .redq import REDQSAC, REDQSOP
from .sac import SAC
from .sop import SOP
from .td3 import TD3


NAMED_ALGORITHMS = {
    'IQL': IQL,
    'MBPO': MBPO,
    'PPO': PPO,
    'REDQSAC': REDQSAC, 'REDQSOP': REDQSOP,
    'SAC': SAC,
    'SOP': SOP,
    'TD3': TD3
}
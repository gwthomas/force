import gymnasium as gym

from .torch_wrapper import TorchWrapper, TorchVectorWrapper
from .util import EnvConfig, VectorEnvConfig, get_env


# Register modified envs
gym.register(
    id="AntMod-v2",
    entry_point="force.env.mujoco:AntModEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0
)

gym.register(
    id="HumanoidMod-v2",
    entry_point="force.env.mujoco:HumanoidModEnv",
    max_episode_steps=1000
)
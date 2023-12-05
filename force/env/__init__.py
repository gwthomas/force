import gymnasium as gym


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


# Register D4RL envs (if installed)
try:
    import d4rl
except:
    print('Failed to import d4rl')
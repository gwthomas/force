import gymnasium as gym

from force.nn.util import torchify, numpyify, get_device


class TorchWrapper(gym.Wrapper):
    def __init__(self, env, device=None):
        super().__init__(env)
        self.device = get_device(device)

    def reset(self):
        return torchify(self.env.reset(), device=self.device)

    def step(self, action):
        action_np = numpyify(action)
        assert self.env.action_space.contains(action_np)
        observation, reward, terminated, truncated, info = self.env.step(action_np)
        return torchify(observation, device=self.device), float(reward), terminated, truncated, info


class TorchVectorWrapper(gym.vector.VectorEnv):
    def __init__(self, vec_env, device=None):
        self.vec_env = vec_env
        self.device = get_device(device)

    @property
    def num_envs(self):
        return self.vec_env.num_envs

    @property
    def observation_space(self):
        return self.vec_env.observation_space

    @property
    def action_space(self):
        return self.vec_env.action_space

    @property
    def single_observation_space(self):
        return self.vec_env.single_observation_space

    @property
    def single_action_space(self):
        return self.vec_env.single_action_space

    @property
    def spec(self):
        return self.vec_env.spec

    def reset(self):
        return torchify(self.vec_env.reset(), device=self.device)

    def step(self, actions):
        actions_np = numpyify(actions)
        assert self.vec_env.action_space.contains(actions_np)
        retvals = list(self.vec_env.step(actions_np))
        for i in range(len(retvals)):
            try:
                retvals[i] = torchify(retvals[i])
            except:
                pass
        return tuple(retvals)
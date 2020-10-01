from gym.envs.mujoco import HalfCheetahEnv as GymHalfCheetahEnv
import numpy as np

from .mujoco_wrapper import MujocoWrapper


class HalfCheetahEnv(GymHalfCheetahEnv, MujocoWrapper):
    @staticmethod
    def done(states):
        return np.zeros(len(states), dtype=bool)

    def qposvel_from_obs(self, obs):
        qpos = np.zeros(9)
        qpos[1:] = obs[:8]
        qvel = obs[8:]
        return qpos, qvel
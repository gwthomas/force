from gym.envs.mujoco import SwimmerEnv as GymSwimmerEnv
import numpy as np

from .mujoco_wrapper import MujocoWrapper


class SwimmerEnv(GymSwimmerEnv, MujocoWrapper):
    @staticmethod
    def done(states):
        return np.zeros(len(states), dtype=bool)

    def qposvel_from_obs(self, obs):
        raise NotImplementedError
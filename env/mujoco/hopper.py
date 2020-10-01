from gym.envs.mujoco.hopper import HopperEnv as GymHopperEnv
import numpy as np

from .mujoco_wrapper import MujocoWrapper

class HopperEnv(GymHopperEnv, MujocoWrapper):
    @staticmethod
    def done(states):
        heights, angs = states[:,0], states[:,1]
        return ~(np.isfinite(states).all(axis=1) & (np.abs(states[:,1:]) < 100).all(axis=1) & (heights > .7) & (np.abs(angs) < .2))

    def qposvel_from_obs(self, obs):
        qpos = np.zeros(6)
        qpos[1:] = obs[:5]
        qvel = obs[5:]
        return qpos, qvel
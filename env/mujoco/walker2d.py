from gym.envs.mujoco.walker2d import Walker2dEnv as GymWalker2dEnv
import numpy as np

from .mujoco_wrapper import MujocoWrapper

class Walker2dEnv(GymWalker2dEnv, MujocoWrapper):
    def _get_obs(self):
        # Override to remove qvel clipping, which messes with the oracle dynamics
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], qvel]).ravel()

    @staticmethod
    def done(states):
        heights, angs = states[:,0], states[:,1]
        return ~((heights > 0.8) & (heights < 2.0) & (angs > -1.0) & (angs < 1.0))

    def qposvel_from_obs(self, obs):
        qpos = np.empty(9)
        qpos[0] = 0.
        qpos[1:] = obs[:8]
        qvel = obs[8:]
        return qpos, qvel
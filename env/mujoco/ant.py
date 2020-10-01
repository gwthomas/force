from gym.envs.mujoco.ant import AntEnv as GymAntEnv
import numpy as np

from .mujoco_wrapper import MujocoWrapper

class AntEnv(GymAntEnv, MujocoWrapper):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    @staticmethod
    def done(states):
        return ~(np.isfinite(states).all(axis=1) & (states[:,0] >= 0.2) & (states[:,0] <= 1.0))

    def qposvel_from_obs(self, obs):
        qpos = np.zeros(15)
        qpos[2:] = obs[:13]
        qvel = obs[13:27]
        return qpos, qvel
from gym.envs.mujoco.humanoid import HumanoidEnv as GymHumanoidEnv
import numpy as np

from .mujoco_wrapper import MujocoWrapper

class HumanoidEnv(GymHumanoidEnv, MujocoWrapper):
    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               # data.cinert.flat,
                               # data.cvel.flat,
                               # data.qfrc_actuator.flat,
                               # data.cfrc_ext.flat
                               ])

    @staticmethod
    def done(states):
        qpos2 = states[:,0]
        return (qpos2 < 1.0) | (qpos2 > 2.0)

    def qposvel_from_obs(self, obs):
        qpos = np.zeros(24)
        qpos[2:] = obs[:22]
        qvel = obs[22:45]

        obs = self._get_obs()
        if not np.all(obs[:22] == self.sim.data.qpos[2:]):
            print('Wrong qpos')
        if not np.all(obs[22:45] == self.sim.data.qvel):
            print('Wrong qvel')
        return qpos, qvel
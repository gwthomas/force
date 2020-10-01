import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahVelJumpEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, goal_vel=3.0):
        self._goal_vel = goal_vel
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    @staticmethod
    def done(states):
        return np.zeros(len(states), dtype=bool)

    def step(self, action):
        if not hasattr(self, "init_zpos"):
            self.init_zpos = self.sim.data.get_body_xpos('torso')[2]
        # self.render()
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward_run = min(reward_run, self._goal_vel)
        reward_jump = 15.0*(self.sim.data.get_body_xpos('torso')[2] - self.init_zpos) #zpos
        reward = reward_ctrl + reward_run + reward_jump
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl, reward_jump=reward_jump)

    def step_obs_act(self, obs, action):
        next_obs = []
        for i in range(obs.shape[0]):
            qpos, qvel = obs[i, :self.sim.data.qpos.shape[0]], obs[i, self.sim.data.qpos.shape[0]+1:]
            self.set_state(qpos, qvel)
            # self.render()
            self.do_simulation(action[i], self.frame_skip)
            ob = self._get_obs()
            next_obs.append(ob)
        return np.array(next_obs)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            # self.sim.data.qpos.flat,
            [self.sim.data.get_body_xpos('torso')[2]],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.init_zpos = self.sim.data.get_body_xpos('torso')[2]
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

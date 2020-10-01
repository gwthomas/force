"""Implementation of Adams et al.'s ODE simulator for HIV treatment.

Adams, Brian Michael, et al. Dynamic multidrug therapies for HIV: Optimal and
STI control approaches. North Carolina State University. Center for Research in
Scientific Computation, 2004. APA.

https://pdfs.semanticscholar.org/c030/127238b1dbad2263fba6b64b5dec7c3ffa20.pdf
"""
import numpy as np

import whynot.gym as wng
from whynot.gym import spaces
from whynot.gym.envs import ODEEnvBuilder, register
from whynot.simulators.hiv import Config, Intervention, simulate, State


def get_intervention(action, time):
    """Return the intervention in the simulator required to take action."""
    return Intervention(time=time, epsilon_1=action[0], epsilon_2=action[1])


def get_reward(intervention, state):
    """Compute the reward based on the observed state and choosen intervention."""
    Q, R1, R2, S = 0.1, 20000.0, 20000.0, 1000.0
    reward = S * state.immune_response - Q * state.free_virus
    reward -= R1 * (intervention.updates["epsilon_1"] ** 2)
    reward -= R2 * (intervention.updates["epsilon_2"] ** 2)
    return reward


def observation_space():
    """Return observation space.

    The state is (uninfected_T1, infected_T1, uninfected_T2, infected_T2,
    free_virus, immune_response) in units (cells/ml, cells/ml, cells/ml,
    cells/ml, copies/ml, cells/ml).
    """
    state_dim = State.num_variables()
    state_space_low = np.zeros(state_dim)
    state_space_high = np.inf * np.ones(state_dim)
    return spaces.Box(state_space_low, state_space_high, dtype=np.float64)


_ContinuousHIVEnv = ODEEnvBuilder(
    simulate_fn=simulate,
    config=Config(),
    initial_state=State(),
    action_space=spaces.Box(low=np.array([0.,0.]), high=np.array([0.7, 0.3])),
    observation_space=observation_space(),
    timestep=1.0,
    intervention_fn=get_intervention,
    reward_fn=get_reward,
)


from gym import Wrapper


class HIVEnv(Wrapper):
    @staticmethod
    def done(states):
        return np.zeros(len(states), dtype=bool)

    def __init__(self):
        super().__init__(_ContinuousHIVEnv())

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), self.reward(reward), done, info

    def observation(self, observation):
        return np.log(observation)

    def reward(self, reward):
        return 1e-6 * reward
from .batch import StatefulBatchedEnv

from force.torch_util import torchify, numpyify


def _get_done(env):
    if hasattr(env.__class__, 'done'):
        return env.__class__.done
    else:
        return _get_done(env.env)


class BatchedVirtualEnv(StatefulBatchedEnv):
    _state_readonly = False

    def __init__(self, env_factory, dynamics_model, n_envs, reward_function=None):
        super().__init__(env_factory(), n_envs)
        self.dynamics_model = dynamics_model
        self.reward_function = reward_function
        self.env_done = _get_done(self.proto_env)

    def termination_function(self, states):
        return torchify(self.env_done(numpyify(states)), to_device=True)

    def _reset_index(self, index):
        # Only one proto env
        return self.proto_env.reset()

    def _step(self, actions):
        next_states, rewards = self.dynamics_model(self._states, actions)
        if self.reward_function is not None:
            rewards = self.reward_function(self._states, actions, next_states)
        dones = self.termination_function(next_states)
        return next_states, rewards, dones, {}
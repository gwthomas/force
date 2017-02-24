import numpy as np
import tensorflow as tf

from gtml.nn.network import Container
from gtml.nn.opt import Optimizer, mean_squared_error
from gtml.rl.core import rollouts, discounted_returns
from gtml.rl.env import vector_var_for_space
from gtml.util.misc import attrwise_cat


def concat_episodes(episodes):
    return attrwise_cat(episodes, ['observations', 'actions', 'rewards'])


class PolicyGradientMethod(Optimizer):
    def __init__(self, policy, value_fn,
            value_fn_fit_itrs=100,
            batchsize=1,
            reg_value_fn_fit=0.0,
            reg_entropy=0.01,
            update_fn=lasagne.updates.adam):
        self.policy = policy
        self.value_fn = value_fn
        self.value_fn_fit_itrs = value_fn_fit_itrs
        self.batchsize = batchsize
        self.reg_value_fn_fit = reg_value_fn_fit
        network = policy.implementation
        observations_var = network.get_input_var()
        actions_var = vector_var_for_space(policy.env.action_space)
        log_probs_var = policy.get_log_probs_var(actions_var)
        returns_var = T.fvector()
        advantages_var = T.fvector()
        loss_var = -T.mean(log_probs_var * advantages_var)
        input_vars = [observations_var, actions_var, advantages_var]
        if reg_value_fn_fit != 0:
            input_vars.append(returns_var)
            loss_var = loss_var + reg_value_fn_fit * mean_squared_error(value_fn.get_output_var(), returns_var)

        updates = update_fn(loss_var, network.get_param_vars())
        super().__init__(network, input_vars, updates)

    def run(self, num_episodes=1, render=False):
        num_updates = num_episodes // self.batchsize
        for _ in range(num_updates):
            episodes = rollouts(self.policy, self.batchsize, render=render)
            observations, actions, rewards = concat_episodes(episodes)
            returns, advantages = [], []
            for episode in episodes:
                Rt = discounted_returns(episode.rewards, 1)
                returns.extend(Rt)
                advantages.extend(Rt - self.value_fn(episode.observations))
            print('Updating policy')
            inputs = [observations, actions, advantages]
            if self.reg_value_fn_fit != 0:
                inputs.append(returns)
            self.step(*inputs)
            self._network.save_params()
            print('Fitting value_fn')
            self.value_fn.fit(episodes, self.value_fn_fit_itrs)

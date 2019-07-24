import numpy as np
import tensorflow as tf

from force.callbacks import Periodic
from force.rl.env import get_gym_env, integral_dimensionality
from force.rl.policy import CategoricalPolicy, GaussianPolicy
from force.rl.ppo import ProximalPolicyOptimization
from force.rl.sampling import rollout
import force.util as util
from force.workflow.config import *


config_info = Config([
    ConfigItem('env', str, REQUIRED),
    ConfigItem('T', int, REQUIRED)
])


def actor_critic_for_env(env, hidden_dims=[100,100]):
    obs_dim = integral_dimensionality(env.observation_space)
    action_dim = integral_dimensionality(env.action_space)
    policy_net = util.mlp(obs_dim, hidden_dims + [action_dim])
    policy = CategoricalPolicy(policy_net)
    value_fn = util.mlp(obs_dim, hidden_dims + [1])
    return policy, value_fn


def main(exp, cfg):
    env = get_gym_env(cfg['env'])
    policy, value_fn = actor_critic_for_env(env)
    alg = ProximalPolicyOptimization(env, policy, value_fn,
                                     tf.keras.optimizers.Adam, T=cfg['T'])

    def evaluate(n_rollouts=100):
        exp.log('Evaluating...')
        episodes = alg.sampler.rollouts(n_rollouts)
        returns = [episode.total_reward for episode in episodes]
        exp.log('Average return over {} rollouts: {}', len(returns), np.mean(returns))

    alg.add_callback('post-iteration', Periodic(10, evaluate))
    alg.run(n_iterations=100000)


variant_specs = {
    'cartpole': {'env': 'CartPole-v0', 'T': 20}
}

def sbatch_args(spec):
    return {
        'ntasks': 1,
        'cpus-per-task': 1,
        'time': '1-0',
        'mem-per-cpu': '4G'
    }

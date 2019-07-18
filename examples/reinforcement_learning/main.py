import numpy as np
import torch

from gtml.models.basic import mlp
from gtml.rl.env import get_env, integral_dimensionality
from gtml.rl.policy import CategoricalPolicy, GaussianPolicy
from gtml.rl.ppo import ProximalPolicyOptimization
from gtml.rl.sampling import rollout
from gtml.workflow.config import *


config_info = Config([
    ConfigItem('env', str, REQUIRED),
    ConfigItem('T', int, REQUIRED),
    ConfigItem('render', bool, False)
])


def actor_critic_for_env(env, hidden_dims=[100,100]):
    obs_dim = integral_dimensionality(env.observation_space)
    action_dim = integral_dimensionality(env.action_space)
    policy_net = mlp([obs_dim] + hidden_dims + [action_dim])
    policy = CategoricalPolicy(policy_net)
    value_fn = mlp([obs_dim] + hidden_dims + [1])
    return policy, value_fn


def main(exp, cfg):
    env = get_env(cfg['env'], should_render=cfg['render'])
    policy, value_fn = actor_critic_for_env(env)

    def evaluate():
        exp.log('Evaluating...')
        episodes = rollout(env, policy, 100)
        returns = [episode.total_reward for episode in episodes]
        exp.log('Average return: {}', np.mean(returns))

    alg = ProximalPolicyOptimization(env, policy, value_fn, torch.optim.Adam,
                                     T=cfg['T'])
    alg.add_callback('post-iteration', evaluate)
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

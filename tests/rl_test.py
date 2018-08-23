from __future__ import print_function

import gym
import torch; nn = torch.nn

import gtml.nn.proto as proto
from gtml.rl.engine import Engine
from gtml.rl.env import Environment, AtariEnvironment, integral_dimensionality
from gtml.rl.policy import CategoricalPolicy, ActorCritic
from gtml.rl.ppo import ProximalPolicyOptimization


def factory(env):
    obs_space_dim = integral_dimensionality(env.observation_space)
    action_space_dim = integral_dimensionality(env.action_space)
    if isinstance(env, AtariEnvironment):
        hidden = 512
        common = proto.atari()
    else:
        hidden = 100
        common = proto.mlp([obs_space_dim, hidden])
    net = nn.Sequential(common, nn.Linear(hidden, action_space_dim))
    value_fn = nn.Sequential(common, nn.Linear(hidden, 1))
    policy = CategoricalPolicy(net)
    return ActorCritic(policy, value_fn)

def print_return(engine, episode):
    k = 10
    if len(engine.episodes) == k:
        rewards = [episode.rewards.sum() for episode in engine.episodes.recent(k)]
        avg_reward = float(torch.mean(torch.tensor(rewards)))
        print('Average reward on last', k, 'episodes:', avg_reward)
        engine.episodes.clear()

def saver(model, path):
    def save(engine):
        torch.save(model.state_dict(), path)
    return save

def go(envname, render, test):
    basename = envname.split('-')[0]
    if basename in ['Breakout']:
        env = AtariEnvironment(envname)
    else:
        env = Environment(envname)
    ac = factory(env)

    policy = ac.policy
    policy_path = envname + '-policy'
    try:
        policy.net.load_state_dict(torch.load(policy_path))
    except:
        print('Failed to load policy for environment', envname)

    value_fn = ac.value_fn
    value_fn_path = envname + '-value-fn'
    try:
        value_fn.load_state_dict(torch.load(value_fn_path))
    except:
        print('Failed to load value function for environment', envname)

    engine = Engine(env, render=render)
    engine.add_callback('post-episode', print_return)
    if test:
        print(engine.evaluate(policy, num_episodes=10))
    else:
        ppo = ProximalPolicyOptimization(ac)
        ppo.add_callback('post-epoch', saver(policy.net, policy_path))
        ppo.add_callback('post-epoch', saver(value_fn, value_fn_path))
        ppo.run(engine, int(1e6))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', metavar='ENVNAME', type=str,
            help='name of the Gym environment to run the agent in')
    parser.add_argument('--render', action='store_true', help='render the environment')
    parser.add_argument('--test', action='store_true', help='test a saved policy')
    parser.add_argument('--load', action='store_true', help='load a saved policy (for further training)')
    args = parser.parse_args()

    go(args.envname, args.render, args.test)

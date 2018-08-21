import gym
import torch

import gtml.nn.proto as proto
from gtml.rl.engine import Engine
from gtml.rl.env import Environment, AtariEnvironment, integral_dimensionality
from gtml.rl.policy import CategoricalPolicy, ActorCritic
from gtml.rl.ppo import ProximalPolicyOptimization


def factory(env):
    basename = env.name.split('-')[0]
    obs_space_dim = integral_dimensionality(env.observation_space)
    action_space_dim = integral_dimensionality(env.action_space)
    if basename in ['Breakout']:
        net = proto.AtariNet(action_space_dim)
    else:
        net = proto.MLP([obs_space_dim, 100, action_space_dim])
    if env.discrete_actions:
        policy = CategoricalPolicy(net)
    else:
        policy = GaussianPolicy(net)
    # valuefn = torch.nn.Linear(obs_space_dim, 1)
    valuefn = proto.MLP([obs_space_dim, 100, 1])
    # valuefn = proto.LinearFunction(net)
    return ActorCritic(policy, valuefn)

def print_return(engine, episode):
    k = 10
    if len(engine.episodes) == k:
        rewards = [episode.rewards.sum() for episode in engine.episodes.recent(k)]
        avg_reward = float(torch.mean(torch.tensor(rewards)))
        print('Average reward on last', k, 'episodes:', avg_reward)
        engine.episodes.clear()

def train(envname, render):
    env = Environment(envname)
    engine = Engine(env, render=render)
    engine.add_callback('post-episode', print_return)
    ac = factory(env)
    ppo = ProximalPolicyOptimization(ac)
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

    if args.test:
        test(args.envname, args.render)
    else:
        train(args.envname, args.render)

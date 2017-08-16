import gym
import numpy as np
import tensorflow as tf

from gtml.common.tf import init_sess
import gtml.nn.layers as layers
import gtml.nn.proto as proto
from gtml.rl.engine import Engine
from gtml.rl.env import Environment, AtariEnvironment
from gtml.rl.policy import SoftmaxPolicy
from gtml.rl.ppo import ProximalPolicyOptimization


def factory(env, observations_in, variable_manager):
    basename = env.name.split('-')[0]
    if basename in ['Breakout']:
        conv_out = proto.convnet(observations_in,
                filters=[(32, 8), (64, 4), (64, 3)],
                strides=[4, 2, 1],
                variable_manager=variable_manager)
        mlp_in = layers.flatten(conv_out)
        hidden_layers = layers.dense(mlp_in, 'dense', 512, tf.nn.relu, variable_manager=variable_manager)
    else:
        hidden_layers = proto.mlp(observations_in, [64, 64], variable_manager=variable_manager, output_nl=tf.nn.relu)
    logits = layers.dense(hidden_layers, 'logits', env.action_space.n, None)
    value_fn = tf.squeeze(layers.dense(hidden_layers, 'value', 1, None))  # squeeze makes it a scalar
    return SoftmaxPolicy(observations_in, logits), value_fn

def print_return(engine, episode):
    print(episode.total_reward)

def train(envname, render):
    env = Environment(envname)
    engine = Engine(env, render=render)
    engine.add_callback('post-episode', print_return)
    ppo = ProximalPolicyOptimization(env, factory)
    with tf.Session():
        init_sess()
        ppo.run(engine, n_iterations=int(1e6))


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

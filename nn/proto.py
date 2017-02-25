import gym
import tensorflow as tf

import gtml.config as cfg
from gtml.nn.layers import *
from gtml.rl.env import integral_dimensionality

def mlp(sizes, input=None, nl=tf.nn.relu, output_nl=None):
    if input is None:
        input = tf.placeholder(cfg.FLOAT_T, shape=[None, sizes[0]], name="mlp_in")

    sofar = input
    for size in sizes[:-1]:
        sofar = DenseLayer(sofar, size, nl)
    sofar = DenseLayer(sofar, sizes[-1], output_nl)

    return sofar

def mlp_for_env(env, hidden_sizes, nl=tf.nn.relu):
    n_input = integral_dimensionality(env.observation_space)
    n_output = integral_dimensionality(env.action_space)
    output_nl = tf.nn.softmax if isinstance(env.action_space, gym.spaces.Discrete) else None
    return mlp([n_input] + hidden_sizes + [n_output], nl=nl, output_nl=output_nl)


def convnet(input_shape, filters, strides, dense_sizes,
        poolings=None,
        input=None,
        conv_nl=tf.nn.relu,
        dense_nl=tf.nn.relu,
        output_nl=None
):
    assert len(input_shape) == 3
    if input is None:
        input = tf.placeholder(cfg.FLOAT_T, shape=[None]+list(input_shape), name="conv_in")

    poolings = [None]*len(filters) if poolings is None else poolings

    sofar = input
    for filter, stride, pooling in zip(filters, strides, poolings):
        n_filters, filter_size = filter
        sofar = ConvLayer(sofar, n_filters, filter_size,
                stride=stride,
                nonlinearity=conv_nl)
        if pooling is not None:
            sofar = MaxPoolLayer(sofar, pooling)

    return mlp(sizes=dense_sizes, input=sofar, nl=dense_nl, output_nl=output_nl)

def dqn_atari(env, output_nl=None, m=4):
    return convnet((m, 84, 84),
            filters=[(32, 8), (64, 4), (64, 3)],
            strides=[4, 2, 1],
            dense_sizes=[512, env.action_space.n],
            output_nl=output_nl)


def shared_value_function(layer):
    return DenseLayer(layer.get_input(), 1, None)

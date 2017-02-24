import gym
import tensorflow as tf

from gtml.nn.layers import *
import gtml.config as cfg


def mlp_proto(sizes, input=None, nl=tf.nn.relu, output_nl=None):
    if input is None:
        input = tf.placeholder(cfg.FLOAT_T, shape=[None, sizes[0]])

    sofar = input
    for size in sizes:
        sofar = DenseLayer(sofar, size, nl)
    sofar = DenseLayer(sofar, sizes[-1], output_nl)

    return sofar

def mlp_proto_for_env(env, hidden_sizes, nl=tf.nn.relu):
    n_input = integral_dimensionality(env.observation_space)
    n_output = integral_dimensionality(env.action_space)
    output_nl = tf.nn.softmax if isinstance(env.action_space, gym.spaces.Discrete) else None
    return mlp_proto([n_input] + hidden_sizes + [n_output], nl=nl, output_nl=output_nl)


def convnet_proto(input_shape, num_out, filters, strides,
        poolings=None,
        input=None,
        conv_nl=tf.nn.relu,
        hidden_sizes=[],
        hidden_nl=tf.nn.relu,
        output_nl=None
):
    assert len(input_shape) == 3
    if input is None:
        input = tf.placeholder(cfg.FLOAT_T, shape=[None]+list(input_shape))

    sofar = input
    for filter, stride, pooling in zip(filters, strides, poolings):
        n_filters, filter_size = filter
        sofar = ConvLayer(sofar, n_filters, filter_size,
                stride=stride,
                nonlinearity=conv_nl)
        if pooling is not None:
            sofar = MaxPoolLayer(sofar, pooling)

    return mlp_proto(sizes=hidden_sizes+[num_out], input=sofar, nl=hidden_nl, output_nl=output_nl)

def dqn_atari_proto(env, output_nl=None, m=4):
    return convnet_proto((m, 84, 84), env.action_space.n,
            filters=[(32, 8), (64, 4), (64, 3)],
            strides=[4, 2, 1],
            hidden_sizes=[512],
            output_nl=output_nl)

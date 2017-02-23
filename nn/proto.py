import gym
import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL

from gtml.rl.env import integral_dimensionality

def mlp_proto(sizes, input=None, nl=NL.rectify, output_nl=None):
    if isinstance(input, lasagne.layers.Layer):
        l_prev = input
        input_var = None
    else:
        if input is None:
            input = T.fmatrix()

        input_var = input
        l_prev = L.InputLayer(
            shape=(None, sizes[0]),
            input_var=input_var
        )

    for size in sizes[1:-1]:
        l_prev = L.DenseLayer(l_prev,
            num_units=size,
            nonlinearity=nl
        )

    l_output = L.DenseLayer(l_prev,
        num_units=sizes[-1],
        nonlinearity=output_nl
    )

    return l_output, input_var


def mlp_proto_for_env(env, hidden_sizes, input_var=None, nl=NL.rectify):
    n_input = integral_dimensionality(env.observation_space)
    n_output = integral_dimensionality(env.action_space)
    output_nl = NL.softmax if isinstance(env.action_space, gym.spaces.Discrete) else None
    return mlp_proto([n_input] + hidden_sizes + [n_output], input_var, nl, output_nl)


def convnet_proto(input_shape, num_out, filters, strides,
        poolings=None,
        input=None,
        conv_nl=NL.rectify,
        hidden_sizes=[100],
        hidden_nl=NL.rectify,
        output_nl=None
):
    assert len(input_shape) == 3

    if isinstance(input, lasagne.layers.Layer):
        l_prev = input
        input_var = None
    else:
        if input is None:
            input = T.ftensor4()

        input_var = input
        l_prev = L.InputLayer(
            shape=(None,) + input_shape,
            input_var=input
        )

    if poolings is None:
        poolings = [None] * len(filters)

    for filter, stride, pooling in zip(filters, strides, poolings):
        n_filters, filter_size = filter
        l_prev = L.Conv2DLayer(l_prev, n_filters, filter_size,
                stride=stride,
                nonlinearity=conv_nl)
        if pooling is not None:
            l_prev = L.MaxPool2DLayer(l_prev, pooling)

    for size in hidden_sizes:
        l_prev = L.DenseLayer(l_prev,
            num_units=size,
            nonlinearity=hidden_nl
        )

    l_output = L.DenseLayer(l_prev,
        num_units=num_out,
        nonlinearity=output_nl
    )

    return l_output, input_var


def dqn_atari_proto(env, output_nl=None, m=4):
    return convnet_proto((m, 84, 84), env.action_space.n,
            filters=[(32, 8), (64, 4), (64, 3)],
            strides=[4, 2, 1],
            hidden_sizes=[512],
            output_nl=output_nl)


def shared_value_function_proto(proto):
    layers = L.get_all_layers(proto)
    return L.DenseLayer(layers[-2], 1, nonlinearity=None)

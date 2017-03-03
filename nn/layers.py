import numpy as np
import tensorflow as tf

from gtml.util.tf import flatten, selection_slice


def _get_tf(o):
    if isinstance(o, tf.Tensor):
        return o
    elif isinstance(o, Layer):
        return o.get_output()
    else:
        raise RuntimeError

def _get_shape(o):
    return _get_tf(o).get_shape().as_list()


class Layer:
    # input may be either a tf tensor or another Layer instance
    def __init__(self, input, output, param_vars):
        self._input = input
        self._output = output
        self._param_vars = param_vars
        self.name = None

    def get_input(self):
        return self._input

    def get_output(self):
        return self._output

    def get_output_shape(self):
        return _get_shape(self._output)

    def get_orig_input(self):
        if isinstance(self._input, tf.Tensor):
            return self._input
        else:
            return self._input.get_orig_input()

    def get_param_vars(self):
        return self._params_vars

    def get_all_param_vars(self):
        if isinstance(self._input, tf.Tensor):
            return self._param_vars
        else:
            return self._input.get_all_param_vars() + self._param_vars

    def get_all_layers(self):
        if isinstance(self._input, tf.Tensor):
            return [self._input, self]
        else:
            return self._input.get_all_layers() + [self]

    def eval(self, feed_dict, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        return sess.run(self._output, feed_dict=feed_dict)

    def __call__(self, input):
        return self.eval({self.get_orig_input(): input})


class DenseLayer(Layer):
    def __init__(self, input, size, nonlinearity, num_leading_axes=1,
            W_init=tf.random_normal_initializer(), b_init=tf.zeros_initializer()):
        input_tf = _get_tf(input)
        flattened_input = flatten(input_tf, num_leading_axes=num_leading_axes)
        input_size = _get_shape(flattened_input)[1]
        self._W = tf.Variable(W_init([input_size, size]))
        self._b = tf.Variable(b_init([size]))
        pre_activation = tf.matmul(flattened_input, self._W) + self._b
        output = nonlinearity(pre_activation) if nonlinearity is not None else pre_activation
        super().__init__(input, output, [self._W, self._b])


class ConvLayer(Layer):
    def __init__(self, input, num_filters, filter_size, stride, nonlinearity, padding='SAME',
            W_init=tf.random_normal_initializer(), b_init=tf.zeros_initializer()):
        input_tf = _get_tf(input)
        input_shape = _get_shape(input_tf)
        in_channels = input_shape[-1]
        out_channels = num_filters
        self._W = tf.Variable(W_init([filter_size, filter_size, in_channels, out_channels]))
        self._b = tf.Variable(b_init([out_channels]))
        self._params = [self._W, self._b]
        pre_activation = tf.nn.conv2d(input_tf, self._W,
                strides=[1, stride, stride, 1], padding=padding) + self._b
        output = nonlinearity(pre_activation) if nonlinearity is not None else pre_activation
        super().__init__(input, output, [self._W, self._b])


class MaxPoolLayer(Layer):
    def __init__(self, input, size, stride=None, padding='SAME'):
        if stride is None:
            stride = size
        input_tf = _get_tf(input)
        output = tf.nn.max_pool(input_tf, ksize=[1, size, size, 1],
                        strides=[1, stride, stride, 1], padding=padding)
        super().__init__(input, output, [])


# For implementing deterministic policies on discrete action spaces
def ArgmaxLayer(Layer):
    def __init__(self, input, axis=1):
        input_tf = _get_tf(input)
        output = tf.argmax(input_tf, axis=axis)
        super().__init__(input, output, [])


# For implementing stochastic policies on discrete action spaces
# Note: assumes that input gives *log* probabilities
class MultinomialLayer(Layer):
    def __init__(self, input, axis=1):
        input_tf = _get_tf(input)
        output = flatten(tf.multinomial(input_tf, 1))
        super().__init__(input, output, [])

    def get_log_probs(self, actions_in, n_in):
        log_probs = _get_tf(self._input)
        return selection_slice(log_probs, actions_in, n_in)

    def get_entropy(self):
        log_probs = _get_tf(self._input)
        return -tf.reduce_sum(log_probs * tf.exp(log_probs))

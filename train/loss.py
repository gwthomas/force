import tensorflow as tf


def l1(x):
    return tf.reduce_sum(tf.abs(x))

def l2(x):
    return tf.reduce_sum(x**2)

def squared_error(y, yhat):
    return tf.reduce_sum((y - yhat)**2)

def mean_squared_error(y, yhat):
    return tf.reduce_mean((y - yhat)**2)

def softmax_cross_entropy(labels, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

def sigmoid_cross_entropy(labels, logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

def policy_gradient_loss(log_probs, advantages):
    return -tf.reduce_sum(log_probs * advantages)


class Loss:
    def __init__(self, terms=None):
        self._terms = [] if terms is None else terms
        self._tf = None

    def loss_vars(self):
        return [loss for c, loss, n in self._terms]

    def names(self):
        return [name for c, l, name in self._terms]

    @property
    def tf(self):
        if self._tf is None:
            for coeff, loss, name in self._terms:
                summand = loss if coeff is None else coeff * loss
                self._tf = summand if self._tf is None else self._tf + summand
        return self._tf

    def add(self, loss, coeff=None, name=None):
        assert self._tf is None  # can't add more terms after getting
        self._terms.append((coeff, loss, name))
        return self # for chaining

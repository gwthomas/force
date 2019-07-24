import datetime

import numpy as np
import tensorflow as tf

from gtml.constants import DEFAULT_TIMESTAMP_FORMAT, NP_FLOAT_TYPE


def npf(x):
    if isinstance(x, np.ndarray):
        return x.astype(NP_FLOAT_TYPE)
    elif isinstance(x, tf.Tensor):
        return x.numpy().astype(NP_FLOAT_TYPE)
    else:
        return np.array(x, dtype=NP_FLOAT_TYPE)

def one_hot(labels):
    n = len(np.unique(labels))
    return np.eye(n)[labels]

def timestamp(format_string=DEFAULT_TIMESTAMP_FORMAT):
    now = datetime.datetime.now()
    return now.strftime(format_string)

def set_random_seed(seed, env=None):
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    if env is not None:
        env.seed(seed)

def mlp(in_dim, sizes, hidden_activation='relu', output_activation=None):
    from tensorflow.keras import layers as L
    model = tf.keras.Sequential()
    model.add(L.Dense(sizes[0], input_shape=(in_dim,), activation=hidden_activation))
    for size in sizes[1:-1]:
        model.add(L.Dense(size, activation=hidden_activation))
    model.add(L.Dense(sizes[-1], activation=output_activation))
    return model

def clone_model(model):
    cloned_model = tf.keras.models.clone_model(model)
    cloned_model.set_weights(model.get_weights())
    return cloned_model

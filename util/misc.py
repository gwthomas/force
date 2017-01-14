import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T
import gtml.config as cfg
from .file import logpath

def superhash(obj):
    if isinstance(obj, np.ndarray):
        array = obj.copy()
        array.flags.writeable = False
        return hash(array.data)
    else:
        return hash(obj)

def safelog(x, epsilon=cfg.DEFAULT_EPSILON):
    return np.log(x + epsilon)

def one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]

def safelog_sym(x, epsilon=cfg.DEFAULT_EPSILON):
    return T.log(x + epsilon)

def add_dim(array, axis=0):
    shape = list(array.shape)
    shape.insert(axis, 1)
    return array.reshape(shape)

def sanity_check_params(params):
    for param in params:
        assert not np.any(np.isnan(param))

# Intended to manufacture callbacks
def param_logger(parametric, filename='params.npz', frequency=1):
    def save(engine, _=None):
        if engine.itr % frequency == 0:
            parametric.save_params(logpath(filename))
    return save

def load_params(parametric, filename='params.npz'):
    path = logpath(filename)
    try:
        parametric.load_params(path)
    except:
        raise RuntimeError('Failed to load network weights from path {}'.format(path))

def keywise(dicts, keys):
    return [np.array([d[key] for d in dicts]) for key in keys]

def attrwise(objects, keys):
    return [np.array([getattr(o, key) for o in objects]) for key in keys]

def keywise_cat(dicts, keys):
    return [np.concatenate([d[key] for d in dicts]) for key in keys]

def attrwise_cat(objects, keys):
    return [np.concatenate([getattr(o, key) for o in objects]) for key in keys]

def show_matrices(ms, cmap=None):
    for m in ms:
        plt.figure()
        plt.imshow(m, cmap)
    plt.show()

def show_matrix(m, cmap=None):
    show_matrices([m], cmap)

def pickle_copy(o):
    return pickle.loads(pickle.dumps(o))

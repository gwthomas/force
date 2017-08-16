import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from gtml.defaults import EPSILON


def safelog(x, epsilon=EPSILON):
    return np.log(x + epsilon)

def one_hot(labels):
    n = len(np.unique(labels))
    return np.eye(n)[labels]

def add_dim(array, axis=0):
    shape = list(array.shape)
    shape.insert(axis, 1)
    return array.reshape(shape)

# Flatten, then concatenate
def conflattenate(arrays):
    return np.concatenate([array.flatten() for array in arrays])

def luminance(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    return 0.2126*r + 0.7152*g + 0.0722*b

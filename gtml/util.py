import torch

from gtml.defaults import EPSILON


def safelog(x, epsilon=EPSILON):
    return torch.log(x + epsilon)

def one_hot(labels):
    n = len(torch.unique(labels))
    return torch.eye(n)[labels]

# Flatten, then concatenate
# def conflattenate(arrays):
    # return torch.concatenate([array.flatten() for array in arrays])

def luminance(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    return 0.2126*r + 0.7152*g + 0.0722*b

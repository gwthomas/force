import numpy as np
import gtml.config as cfg

def discounted_partial_sums(rewards, discount):
    return np.cumsum(np.array(rewards) * discount**np.arange(len(rewards)))

def discounted_sum(rewards, discount):
    return np.sum(np.array(rewards) * discount**np.arange(len(rewards)))
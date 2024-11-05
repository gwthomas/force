from .module import *
from .layers import *
from .optim import Optimizer

# Set number of threads PyTorch may use
import torch
from force.defaults import TORCH_NUM_THREADS
torch.set_num_threads(TORCH_NUM_THREADS)
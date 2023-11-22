import torch

PRECISION = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_LAYERS = 2
HIDDEN_DIM = 256
ACTIVATION = 'relu'
OPTIMIZER_TYPE = 'Adam'
LEARNING_RATE = 0.001
BATCH_SIZE = 128
DISCOUNT = 0.99
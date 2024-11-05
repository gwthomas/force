import torch

DATETIME_FORMAT = '%y-%m-%d_%H.%M.%S'
PRECISION = 2   # for printing
NUMERICAL_EPSILON = 1e-6  # small value for numerical purposes
MAX_SEED = 100
TORCH_NUM_THREADS = 1
HIDDEN_LAYERS = 2
HIDDEN_DIM = 256
ACTIVATION = 'relu'
OPTIMIZER_TYPE = 'Adam'
LEARNING_RATE = 0.001
BATCH_SIZE = 256
BATCH_MAP_SIZE = 1000  # Adjust according to GPU memory limits
DISCOUNT = 0.99

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
DEVICE = torch.device(DEVICE)
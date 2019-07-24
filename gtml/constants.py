import numpy as np

INF = float('inf')

NP_FLOAT_TYPE = np.float32

# Defaults
DEFAULT_EPSILON = 1e-8
DEFAULT_DISCOUNT = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_BATCH_SIZE = 128
DEFAULT_TIMESTAMP_FORMAT = '%m-%d-%y_%H.%M.%S'
DEFAULT_NUM_WORKERS = 2
DEFAULT_FIGURE_SIZE = (8,6)

# Some global variables are to be read from disk so that their values can vary
# from one machine to another without modifying this file.
# In particular, paths often differ across machines.
from configparser import ConfigParser
from pathlib import Path
_gtml_config = ConfigParser()
_gtml_config.read(Path.home() / 'gtml.cfg')
if len(_gtml_config.sections()) == 0:
    print('Failed to read gtml.cfg')
    exit(1)
DATASETS_DIR = _gtml_config['paths']['DatasetsDir']
del _gtml_config

import os
import pickle
import gtml.config as cfg

# Get the path of a file in the log directory
def logpath(filename):
    return os.path.expanduser(os.path.join(cfg.LOG_DIR, filename))

def unpickle(file):
    f = open(file, 'rb')
    retval = pickle.load(f, encoding='latin1')
    f.close()
    return retval

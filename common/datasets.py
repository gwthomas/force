try: import cPickle as pickle
except: import pickle

import numpy as np
import os


def unpickle(file):
    f = open(file, 'rb')
    retval = pickle.load(f, encoding='latin1')
    f.close()
    return retval


def load_cifar(dir):
    def reshape_image(img):
        shape = (32,32)
        r = img[:1024].reshape(shape)
        g = img[1024:2048].reshape(shape)
        b = img[2048:].reshape(shape)
        return np.stack([r,g,b], axis=-1) / 255.0

    Xtrain, Ytrain, Xtest, Ytest = [], [], [], []
    for i in range(1,6):
        batch_dict = unpickle(os.path.join(dir, 'data_batch_%i' % i))
        Xtrain.extend([reshape_image(img) for img in batch_dict['data']])
        Ytrain.extend(batch_dict['labels'])
    Xtrain = np.stack(Xtrain).astype(float)
    Ytrain = np.array(Ytrain)

    test_dict = unpickle(os.path.join(dir, 'test_batch'))
    Xtest = np.stack([reshape_image(img) for img in test_dict['data']])
    Ytest = np.array(test_dict['labels'])

    return Xtrain, Ytrain, Xtest, Ytest


def load_mnist(dir):
    def reshape_images(imgs):
        return imgs.reshape((len(imgs), 28, 28, 1)) / 255.0

    from mnist import MNIST
    mndata = MNIST(dir)
    Xtrain, Ytrain = map(np.array, mndata.load_training())
    Xtest, Ytest = map(np.array, mndata.load_testing())
    Xtrain = reshape_images(Xtrain).astype(float)
    Xtest = reshape_images(Xtest).astype(float)
    return Xtrain, Ytrain, Xtest, Ytest

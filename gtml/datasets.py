try: import cPickle as pickle
except: import pickle

import os
import torch


def unpickle(file, encoding=None):
    f = open(file, 'rb')
    if encoding:
        try:
            retval = pickle.load(f, encoding=encoding)
        except: # in python 2, load() doesn't accept encoding
            retval = pickle.load(f)
    else:
        retval = pickle.load(f)
    f.close()
    return retval


def load_cifar(dir):
    # def process_image(img):
    #     shape = (32,32)
    #     r = img[:1024].reshape(shape)
    #     g = img[1024:2048].reshape(shape)
    #     b = img[2048:].reshape(shape)
    #     return torch.stack([r,g,b]) / 255.0
    def process_images(imgs):
        return imgs.reshape((len(imgs), 3, 32, 32)).float() / 255.0

    names = ['data_batch_%i' % i for i in range(1,6)] + ['test_batch']
    X, Y = [], []
    for name in names:
        batch_dict = unpickle(os.path.join(dir, name), encoding='latin1')
        data = torch.from_numpy(batch_dict['data'])
        labels = torch.tensor(batch_dict['labels'])
        X.append(data)
        Y.append(labels)
    Xtrain, Ytrain = torch.cat(X[:-1]), torch.cat(Y[:-1])
    Xtest, Ytest = X[-1], Y[-1]
    Xtrain = process_images(Xtrain)
    Xtest = process_images(Xtest)
    return Xtrain, Ytrain, Xtest, Ytest

def load_mnist(dir):
    def process_images(imgs):
        return imgs.reshape((len(imgs), 1, 28, 28)).float() / 255.0

    from mnist import MNIST
    mndata = MNIST(dir)
    Xtrain, Ytrain = map(torch.tensor, mndata.load_training())
    Xtest, Ytest = map(torch.tensor, mndata.load_testing())
    Xtrain = process_images(Xtrain)
    Xtest = process_images(Xtest)
    return Xtrain, Ytrain, Xtest, Ytest

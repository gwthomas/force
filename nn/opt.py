import lasagne
import numpy as np
import theano
import theano.tensor as T
import gtml.config as cfg


def squared_error(yhat, y):
    return T.sum((yhat - y)**2)

def mean_squared_error(yhat, y):
    return T.mean((yhat - y)**2)

def cross_entropy_error(yhat, y):
    return T.mean(lasagne.objectives.categorical_crossentropy(yhat, y))


class Optimizer(object):
    def __init__(self, input_vars, updates, loss_var=None):
        self._update = theano.function(
            inputs=input_vars,
            outputs=[loss_var] if loss_var is not None else [],
            updates=updates,
            allow_input_downcast=True
        )

    def step(self, *args):
        return self._update(*args)


class SupervisedLearning(Optimizer):
    def __init__(self, network, loss_fn,
            update_fn=lasagne.updates.adam,
            target_var=None):
        self.network = network
        input_var = network.get_input_var()
        output_var = network.get_output_var()
        output_dim = len(network.get_output_shape())
        if target_var is None:
            target_var = T.TensorType(cfg.FLOAT_T, (False,)*output_dim)()
        loss_var = loss_fn(output_var, target_var)
        updates = update_fn(loss_var, network.get_param_vars())
        super().__init__([input_var, target_var], updates, loss_var)

    def run(self, X, Y, itrs=100, batchsize=None, verbose=False):
        n = len(X)
        if batchsize is None:
            batchsize = n

        losses = []
        for itr in range(itrs):
            batchidx = np.random.permutation(n)[:batchsize]
            loss = self.step(X[batchidx], Y[batchidx])
            losses.append(loss)
            if verbose:
                print('Itr %i: loss = %f' % (itr, loss))

        self.network.save_params()
        return np.array(losses)

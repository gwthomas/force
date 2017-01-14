import lasagne
import numpy as np
import theano
import theano.tensor as T
import gtml.config as cfg

def mean_squared_error(yhat, y):
    return T.mean((yhat - y)**2)

def cross_entropy_error(yhat, y):
    return T.mean(lasagne.objectives.categorical_crossentropy(yhat, y))

class Optimizer(object):
    def __init__(self, network, loss_fn, update_fn, target_var=None):
        self._network = network
        input_var = network.get_input_var()
        output_var = network.get_output_var()
        output_dim = len(network.get_output_shape())
        if target_var is None:
            target_var = T.TensorType(cfg.FLOAT_T, (False,)*output_dim)()
        loss_var = loss_fn(output_var, target_var)
        update_info = update_fn(loss_var, network.get_param_vars())
        self._update = theano.function(
                inputs=[input_var, target_var], outputs=loss_var,
                updates=update_info,
                allow_input_downcast=True
        )

    def run(self, X, Y, itrs=100, batchsize=100, filename='params.npz'):
        n = len(X)
        for itr in range(itrs):
            batchidx = np.random.permutation(n)[:batchsize]
            loss = self._update(X[batchidx], Y[batchidx])
            print('Itr %i: loss = %f' % (itr, loss))
            self._network.save_params(filename)

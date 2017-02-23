import lasagne.layers as L
import numpy as np
import theano


class Network(object):
    def __init__(self, prototype, input_var, name, filename=None):
        self._input_var = input_var
        self._output_var = L.get_output(prototype, inputs=input_var)
        self._output_shape = L.get_output_shape(prototype)
        self._param_vars = L.get_all_params(prototype)
        self._fn = theano.function(
            inputs=[self._input_var],
            outputs=[self._output_var],
            allow_input_downcast=True
        )
        self.name = name
        self.filename = name + '.params' if filename is None else filename

    def get_input_var(self):
        return self._input_var

    def get_output_var(self):
        return self._output_var

    def get_output_shape(self):
        return self._output_shape

    def get_param_vars(self):
        return self._param_vars

    def get_params(self):
        return [np.array(param_var.eval()) for param_var in self.get_param_vars()]

    def set_params(self, params):
        assert len(params) == len(self._param_vars)
        for param_var, new_value in zip(self.get_param_vars(), params):
            param_var.set_value(new_value)

    def save_params(self):
        with open(self.filename, 'wb') as f:
            np.savez(f, *self.get_params())

    def load_params(self):
        with np.load(self.filename) as data:
            self.set_params([data['arr_'+str(i)] for i in range(len(data.files))])

    def try_load_params(self):
        try:
            self.load_params()
            return True
        except:
            return False

    def forward(self, *args):
        return self._fn(*args)[0]

    def __call__(self, *args):
        return self.forward(*args)

    def classification_accuracy(self, X, Y):
        return np.mean(np.argmax(self.forward(X), 1) == Y)


class Container(Network):
    def __init__(self, implementation):
        self.implementation = implementation

    def get_input_var(self):
        return self.implementation.get_input_var()

    def get_output_var(self):
        return self.implementation.get_output_var()

    def get_output_shape(self):
        return self.implementation.get_output_shape()

    def get_param_vars(self):
        return self.implementation.get_param_vars()

    def get_params(self):
        return self.implementation.get_params()

    def set_params(self, params):
        self.implementation.set_params(params)

    def save_params(self):
        self.implementation.save_params()

    def load_params(self):
        self.implementation.load_params()

    def __call__(self, *args):
        return self.implementation(*args)


# A collection of networks, essentially for simplifying shared parameter lists.
# Has no input/output and can't be called
class Collection(Network):
    def __init__(self, networks, name, filename=None):
        self._networks = list(networks)
        self._bookkeeping()
        self.name = name
        self.filename = name + '.params' if filename is None else filename

    def add(self, network):
        self._networks.append(network)
        self._bookkeeping()

    def _bookkeeping(self):
        self._param_vars = []
        for network in self._networks:
            for param_var in network.get_param_vars():
                if param_var not in self._param_vars:
                    self._param_vars.append(param_var)

    def get_input_var(self):
        raise NotImplementedError

    def get_output_var(self):
        raise NotImplementedError

    def get_output_shape(self):
        raise NotImplementedError

    def forward(self, *args):
        raise NotImplementedError

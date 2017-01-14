import lasagne.layers as L
import numpy as np
import theano

class Network(object):
    def __init__(self, prototype, input_var=None):
        self._input_var = input_var
        self._output_var = L.get_output(prototype, inputs=input_var)
        self._output_shape = L.get_output_shape(prototype)
        self._param_vars = L.get_all_params(prototype)
        self._fn = theano.function(
            inputs=[self._input_var],
            outputs=[self._output_var],
            allow_input_downcast=True
        )

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

    def save_params(self, filename):
        with open(filename, 'wb') as f:
            np.savez(f, *self.get_params())

    def load_params(self, filename):
        with np.load(filename) as data:
            self.set_params([data['arr_'+str(i)] for i in range(len(data.files))])

    def forward(self, *args):
        return self._fn(*args)[0]

    def __call__(self, *args):
        return self.forward(*args)

    def classification_accuracy(self, X, Y):
        return np.mean(np.argmax(self.forward(X), 1) == Y)

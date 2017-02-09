import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL

def mlp(sizes,
        input_var=None,
        nl=NL.rectify,
        output_nl=None  # Change to softmax to get probabilities
):
    if input_var is None:
        input_var = T.fmatrix()

    l_prev = L.InputLayer(
        shape=(None, sizes[0]),
        input_var=input_var
    )

    for size in sizes[1:-1]:
        l_prev = L.DenseLayer(l_prev,
            num_units=size,
            nonlinearity=nl
        )

    l_output = L.DenseLayer(l_prev,
        num_units=sizes[-1],
        nonlinearity=output_nl,
        name="output"
    )

    return l_output, input_var

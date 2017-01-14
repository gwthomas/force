import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL

def convnet(input_shape, num_out, filters, poolings,
        input_var=None,
        conv_nl=NL.rectify,
        hidden_sizes=[100],
        hidden_nl=NL.rectify,
        output_nl=None  # Change to softmax to get probabilities
):
    assert len(input_shape) == 3

    if input_var is None:
        input_var = T.ftensor4()

    l_prev = L.InputLayer(
        shape=(None,) + input_shape,
        input_var=input_var
    )

    for filter, pooling in zip(filters, poolings):
        l_prev = L.Conv2DLayer(l_prev, filter[0], filter[1],
                nonlinearity=conv_nl)
        l_prev = L.MaxPool2DLayer(l_prev, pooling)

    for size in hidden_sizes:
        l_prev = L.DenseLayer(l_prev,
            num_units=size,
            nonlinearity=hidden_nl
        )

    l_output = L.DenseLayer(l_prev,
        num_units=num_out,
        nonlinearity=output_nl,
        name="output"
    )

    return l_output, input_var

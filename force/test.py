import numpy as np
import tensorflow as tf


def argmax_prediction(model, x):
    return tf.argmax(model(x), axis=1)

def test(model, dataset, prediction=argmax_prediction, criterion=tf.equal):
    batch_results = []
    for x, y in dataset:
        y_hat = prediction(model, x)
        batch_results.append(criterion(y_hat, y))
    all_results = np.concatenate(batch_results)
    return np.mean(all_results)

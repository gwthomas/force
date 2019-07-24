import tensorflow as tf
import tensorflow_datasets as tfds

from gtml.datasets import mnist
from gtml.train import EpochalMinimizer
from gtml.test import test
import gtml.util as util
from gtml.workflow.config import *

import pdb

config_info = Config([
    ConfigItem('dataset', ('cifar10', 'mnist', 'svhn'), REQUIRED),
    ConfigItem('init_lr', float, 0.1),
    ConfigItem('batch_size', int, 128),
    ConfigItem('weight_decay', float, 1e-4),
    ConfigItem('momentum', float, 0.9),
    ConfigItem('n_epochs', int, 200),
    ConfigItem('eval_train', bool, True)
])


def create_model(in_shape, n_classes):
    from tensorflow.keras import layers as L
    model = tf.keras.Sequential()
    model.add(L.Conv2D(64, (3, 3), input_shape=in_shape, activation='relu'))
    model.add(L.Flatten())
    model.add(L.Dense(128, activation='relu'))
    model.add(L.Dense(n_classes))
    return model

def main(exp, cfg):
    if cfg['dataset'] == 'mnist':
        train_set, test_set = mnist()

    model = create_model(train_set.output_shapes[0], 10)
    def loss_fn(batch):
        x, y = batch
        return tf.losses.sparse_softmax_cross_entropy(y, model(x))
    optimizer = tf.train.MomentumOptimizer(cfg['init_lr'], cfg['momentum'])
    train = EpochalMinimizer(loss_fn, model.weights, optimizer, train_set.batch(cfg['batch_size']))

    exp.setup_checkpointing({
        'model': model,
        'optimizer': optimizer,
        # 'train': train
    })

    def evaluate():
        exp.log('Evaluating...')
        test_acc = test(model, test_set.batch(cfg['batch_size']))
        exp.data['test_acc'].append(test_acc)
        if cfg['eval_train']:
            train_acc = test(model, train_set.batch(cfg['batch_size']))
            exp.data['train_acc'].append(train_acc)
            exp.log('Accuracy: train = {}, test = {}', train_acc, test_acc)
        else:
            exp.log('Accuracy: test = {}', test_acc)

    def post_step_callback(steps_taken, loss):
        exp.log('Step {}, Loss = {}', steps_taken, loss)
        exp.data['loss'].append(loss)

    def post_epoch_callback(epochs_taken):
        exp.log('{} epochs completed.', epochs_taken)
        evaluate()
        exp.save()

    train.add_callback('post_step', post_step_callback)
    train.add_callback('post_epoch', post_epoch_callback)
    train.run(cfg['n_epochs'])


variant_specs = {
    'mnist': {'dataset': 'mnist'}
}

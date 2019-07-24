import tensorflow as tf
import tensorflow_datasets as tfds

from force.constants import DATASETS_DIR


def mnist():
    def process_example(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return (image, label)
    dataset = tfds.load('mnist', data_dir=DATASETS_DIR, as_supervised=True)
    train_set = dataset['train']
    test_set = dataset['test']
    return train_set.map(process_example), test_set.map(process_example)

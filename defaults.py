import tensorflow as tf

BATCHSIZE = 64
EPSILON = 1e-8
INT_T = tf.int64
FLOAT_T = tf.float32
GAE_LAMBDA = 0.98
OPTIMIZER = tf.train.AdamOptimizer()

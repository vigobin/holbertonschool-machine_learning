#!/usr/bin/env python3
"""Create a Layer with Dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using dropout"""
    initializer = tf.initializers.GlorotUniform()
    W = tf.Variable(initializer(shape=(n, prev.shape[1])))
    b = tf.Variable(tf.zeros((n,)))

    if activation == 'tanh':
        activation_fn = tf.nn.tanh
    elif activation == 'softmax':
        activation_fn = tf.nn.softmax

    z = tf.matmul(W, prev) + b
    a = activation_fn(z)
    dropout_mask = tf.cast(tf.random.uniform(
        shape=tf.shape(a)) < keep_prob, dtype=tf.float32)
    a_dropout = tf.multiply(a, dropout_mask) / keep_prob

    return a_dropout

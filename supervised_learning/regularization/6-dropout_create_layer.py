#!/usr/bin/env python3
"""Create a Layer with Dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using dropout"""
    init = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
    dp_mask = tf.layers.Dropout(rate=(1-keep_prob))
    layers = tf.layers.Dense(units=n,
                             activation=activation,
                             kernel_initializer=init,
                             kernel_regularizer=dp_mask)(prev)
    return layers

#!/usr/bin/env python3
"""Batch Normalization Upgraded"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a
    neural network in tensorflow"""
    w_init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.dense(prev, n,
                            kernel_initializer=w_init)
    gamma = tf.Variable(tf.ones(n), trainable=True)
    beta = tf.Variable(tf.ones(n), trainable=True)
    m, var = tf.nn.moments(layer, 0)
    Z = tf.nn.batch_normalization(x=layer, mean=m, variance=var,
                                  offset=beta, scale=gamma,
                                  variance_epsilon=1e-8)
    return Z

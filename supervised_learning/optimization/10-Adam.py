#!/usr/bin/env python3
"""Adam Upgraded"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training operation for a neural network in
    tensorflow using the Adam optimization algorithm"""
    op = tf.train.AdamOptimizer(alpha, beta1=beta1,
                                 epsilon=epsilon).minimize(loss)
    return op

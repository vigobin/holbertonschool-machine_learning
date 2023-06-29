#!/usr/bin/env python3
"""L2 Regularization Cost"""

import tensorflow as tf


def l2_reg_cost(cost, lambtha):
    """calculates the cost of a neural network with L2 regularization"""
    regularizer = tf.reduce_sum(
        [tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    l2_loss = lambtha * regularizer
    total_cost = cost + l2_loss
    return total_cost

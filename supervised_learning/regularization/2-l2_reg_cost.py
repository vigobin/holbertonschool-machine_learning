#!/usr/bin/env python3
"""L2 Regularization Cost"""

import tensorflow as tf


def l2_reg_cost(cost, lambtha):
    """calculates the cost of a neural network with L2 regularization"""
    l2_reg_cost = tf.losses.get_regularization_losses()
    return (cost + l2_reg_cost)

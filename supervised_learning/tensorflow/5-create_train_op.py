#!/usr/bin/env python3
"""Train_Op function"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """creates the training operation for the network"""
    gradient_descent = tf.train.GradientDescentOptimizer(alpha)
    return gradient_descent.minimize(loss)

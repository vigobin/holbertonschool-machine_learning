#!/usr/bin/env python3
"""RMSProp Upgraded"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon): 
    """creates the training operation for a neural network in
    tensorflow using the RMSProp optimization algorithm"""
    op = tf.train.RMSPropOptimizer(alpha, decay=beta2,
    epsilon=epsilon).minimize(loss)
    return op

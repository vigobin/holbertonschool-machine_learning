#!/usr/bin/env python3
"""Loss"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """calculates the softmax cross-entropy loss of a prediction"""
    return tf.losses.softmax_cross_entropy(y, y_pred)

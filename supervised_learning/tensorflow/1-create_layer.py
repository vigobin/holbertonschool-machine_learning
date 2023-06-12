#!/usr/bin/env python3
"""Layers"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """Returns: the tensor output of the layer"""

    initializer= tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.dense(prev, n, activation,
                            kernel_initializer=initializer)
    return layer

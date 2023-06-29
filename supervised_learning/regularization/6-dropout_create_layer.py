#!/usr/bin/env python3
"""Create a Layer with Dropout"""

import numpy as np


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using dropout"""
    W = np.random.randn(n, prev.shape[0]) * np.sqrt(2 / prev.shape[0])
    b = np.zeros((n, 1))

    if activation == 'tanh':
        activation_fn = np.tanh
    elif activation == 'softmax':
        activation_fn = np.softmax
    layer = {
        'W': W,
        'b': b,
        'activation': activation_fn,
        'keep_prob': keep_prob
    }

    return layer

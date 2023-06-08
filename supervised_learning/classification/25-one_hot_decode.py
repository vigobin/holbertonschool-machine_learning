#!/usr/bin/env python3
"""one-Hot Encode"""

import numpy as np


def one_hot_decode(one_hot):
    """converts a one-hot matrix into a vector of labels"""
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None

    vector = one_hot.transpose().argmax(axis=1)
    return vector

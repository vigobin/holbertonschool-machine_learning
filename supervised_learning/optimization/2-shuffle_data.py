#!/usr/bin/env python3
"""Shuffle Data"""

import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way"""
    m = X.shape[0]
    shuffle_pattern = np.random.permutation(m)
    X_shuffled = X[shuffle_pattern]
    Y_shuffled = Y[shuffle_pattern]
    return X_shuffled, Y_shuffled

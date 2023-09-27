#!/usr/bin/env python3
"""Variance function"""

import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance for a data set.
    X is a numpy.ndarray of shape (n, d) containing the data set.
    C is a numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster.
    Returns: var, or None on failure - var is the total variance"""
    if type(X) is not np.ndarray or type(C) is not np.ndarray:
        return None
    if len(C.shape) != 2 or len(X.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    var = np.sum((X - C[:, np.newaxis])**2, axis=-1)
    mean = np.sqrt(var)
    min = np.min(mean, axis=0)
    var = np.sum(np.sum(min ** 2))
    return var

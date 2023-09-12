#!/usr/bin/env python3
"""Mean and Covariance"""

import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance of a data set
        Returns:
            mean: numpy.ndarray of shape (1, d) containing the
                mean of the data set
            cov: numpy.ndarray of shape (d, d) containing the covariance
                matrix of the data set"""
    if type(X) is not np.ndarray:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)

    cov = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            cov[i, j] = np.sum((X[:, i] - mean[:, i]) * (
                X[:, j] - mean[:, j])) / (n - 1)

    return mean, cov

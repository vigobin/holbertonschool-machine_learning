#!/usr/bin/env python3
"""PDF function"""

import numpy as np


def pdf(X, m, S):
    """Calculates the probability density function of a Gaussian distribution.
    X is a numpy.ndarray of shape (n, d) containing the data points whose PDF
        should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of the distribution
    S is a numpy.ndarray of shape (d, d) containing the covariance of the
        distribution
    Returns: P, or None on failure
    P is a numpy.ndarray of shape (n,) containing the PDF values for
        each data point
    All values in P should have a minimum value of 1e-300"""
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0] or X.shape[1] != S.shape[1]:
        return None
    
    d = X.shape[1]

    determinant_S = np.linalg.det(S)
    if determinant_S == 0:
        return None
    
    inverse_S = np.linalg.inv(S)

    difference = X - m

    exponent = -0.5 * np.sum(difference.dot(inverse_S) * difference, axis=1)

    norm_const = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(determinant_S))

    P = norm_const * np.exp(exponent)

    P[P < 1e-300] = 1e-300

    return P

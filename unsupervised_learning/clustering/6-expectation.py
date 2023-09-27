#!/usr/bin/env python3
"""Expectation function"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm for a GMM
    X is a numpy.ndarray of shape (n, d) containing the data set.
    pi is a numpy.ndarray of shape (k,) containing the priors for each cluster.
    m is a numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster.
    S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
        for each cluster.
    Returns: g, li, or None, None on failure
        g is a numpy.ndarray of shape (k, n) containing the posterior
            probabilities for each data point in each cluster.
        li is the total log likelihood."""
    n, d = X.shape
    k = pi.shape[0]

    pdf_values = np.zeros((k, n))
    for i in range(k):
        pdf_values[i] = pdf(X, m[i], S[i])

    g = pi[:, np.newaxis] * pdf_values
    g /= np.sum(g, axis=0, keepdims=True)

    li = np.sum(np.log(np.sum(g, axis=0)))

    return g, li

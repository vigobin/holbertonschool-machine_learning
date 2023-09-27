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
    if type(X) is not np.ndarray or type(m) is not np.ndarray:
        return None, None
    if type(S) is not np.ndarray or type(pi) is not np.ndarray:
        return None, None
    if len(X.shape) != 2 or len(S.shape) != 3:
        return None, None
    if len(m.shape) != 2 or len(pi.shape) != 1:
        return None, None
    if m.shape[1] != X.shape[1]:
        return None, None
    if S.shape[2] != S.shape[1]:
        return None, None
    if S.shape[0] != pi.shape[0] or S.shape[0] != m.shape[0]:
        return None, None
    if np.min(pi) < 0:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    pdf_values = np.zeros((k, n))
    for i in range(k):
        pdf_values[i] = pdf(X, m[i], S[i]) * pi[i]

    g = pi[:, np.newaxis] * pdf_values
    g /= np.sum(g, axis=0, keepdims=True)

    li = np.sum(np.log(np.sum(pdf_values, axis=0)))

    return g, li

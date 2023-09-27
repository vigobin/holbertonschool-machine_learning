#!/usr/bin/env python3
"""Maximization function"""

import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM.
    X is a numpy.ndarray of shape (n, d) containing the data set.
    g is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster.
    Returns: pi, m, S, or None, None, None on failure.
    pi is a numpy.ndarray of shape (k,) containing the updated priors
        for each cluster.
    m is a numpy.ndarray of shape (k, d) containing the updated centroid means
        for each cluster.
    S is a numpy.ndarray of shape (k, d, d) containing the updated covariance
        matrices for each cluster"""
    if type(g) is not np.ndarray or len(g.shape) != 2:
        return (None, None, None)
    if X.shape[0] != g.shape[1]:
        return (None, None, None)
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return (None, None, None)
    try:
        k, n = g.shape
        d = X.shape[1]

        pi = np.sum(g, axis=1) / n

        m = (g @ X) / np.sum(g, axis=1)[:, np.newaxis]
        S = np.zeros((k, d, d))
        for i in range(k):
            diff = X - m[i]
            weighted_diff = (g[i, :, np.newaxis] * diff).T @ diff
            S[i] = weighted_diff / np.sum(g[i])

        return pi, m, S
    except Exception:
        return None, None, None

#!/usr/bin/env python3
"""Initialize GMM"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model:
    Returns: pi, m, S, or None, None, None on failure
    pi is a numpy.ndarray of shape (k,) containing the priors for each
        cluster,initialized evenly.
    m is a numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster, initialized with K-means.
    S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
        for each cluster, initialized as identity matrices."""
    if k <= 0:
        return None, None, None

    n, d = X.shape
    pi = np.full(k, 1 / k)
    centroids, _ = kmeans(X, k)
    m = centroids

    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S

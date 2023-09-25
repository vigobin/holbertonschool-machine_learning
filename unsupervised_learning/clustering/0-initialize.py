#!/usr/bin/env python3
"""Initialize K-means"""

import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means.
    Returns: a numpy.ndarray of shape (k, d) containing the
        initialized centroids for each cluster, or None on failure"""
    if k <= 0:
        return None

    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)

    centroids = np.random.uniform(low=min_values, high=max_values, size=(
        k, X.shape[1]))

    return centroids

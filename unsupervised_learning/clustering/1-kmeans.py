#!/usr/bin/env python3
"""Initialize K-means"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset.
    X is a numpy.ndarray of shape (n, d) containing the dataset
    n is the number of data points
    d is the number of dimensions for each data point.
    k is a positive integer containing the number of clusters.
    iterations: positive integer containing the maximum number of iterations.
    Returns: C, clss, or None, None on failure
    C is a numpy.ndarray of shape (k, d) containing the centroid means
    for each cluster.
    clss: numpy.ndarray of shape (n,) containing the index of the cluster in C
    that each data point belongs to."""

    if type(k) is not int or k <= 0:
        return None, None
    if type(X) is not np.ndarray or X.shape[0] <= k:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    C = initialize(X, k)

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

        clss = np.argmin(distances, axis=1)
        new_centroids = np.copy(C)
        for i in range(k):
            if i not in clss:
                new_centroids[i] = initialize(X, 1)[0]
            else:
                new_centroids[i] = np.mean(X[clss == i], axis=0)

        if np.all(C == new_centroids):
            break

        C = new_centroids

    return C, clss


def initialize(X, k):
    """Initializes cluster centroids for K-means.
    Returns: a numpy.ndarray of shape (k, d) containing the
        initialized centroids for each cluster, or None on failure"""
    if type(k) is not int or k <= 0:
        return None
    if type(X) is not np.ndarray or X.shape[0] <= k:
        return None

    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)

    centroids = np.random.uniform(low=min_values, high=max_values, size=(
        k, X.shape[1]))

    return centroids

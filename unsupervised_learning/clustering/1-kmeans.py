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
    centroids = initialize(X, k)

    if centroids is None:
        return None, None

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.sum(
                labels == i) > 0 else initialize(
                    X, 1)[0] for i in range(k)])

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, labels


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

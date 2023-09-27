#!/usr/bin/env python3
"""Optimize K function"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance.
    Returns: results, d_vars, or None, None on failure
    results is a list containing the outputs of K-means for each cluster size
    d_vars is a list containing the difference in variance from the smallest
        cluster size for each cluster size."""
    if kmax is None:
        kmax = min(len(X), 10)
    if kmin < 1 or kmax < kmin:
        return None, None

    results = []
    d_vars = []

    min_var = None

    for k in range(kmin, kmax + 1):
        centroids, cluster_assignments = kmeans(X, k, iterations)

        if centroids is None:
            return None, None
        total_var = variance(X, centroids)
        results.append((centroids, cluster_assignments))

        if k == kmin:
            min_var = total_var
            d_vars.append(0)
        else:
            d_vars.append(min_var - total_var)

    return results, d_vars

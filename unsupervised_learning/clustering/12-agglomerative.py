#!/usr/bin/env python3
"""Agglomerative clustering"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset:
    X is a numpy.ndarray of shape (n, d) containing the dataset.
    dist is the maximum cophenetic distance for all clusters.
    Performs agglomerative clustering with Ward linkage.
    Displays the dendrogram with each cluster displayed in a different color.
    Returns: clss, a numpy.ndarray of shape (n,) containing the cluster
        indices for each data point"""
    linkage_matrix = scipy.cluster.hierarchy.linkage(X, method='ward')
    dendogram = scipy.cluster.hierarchy.dendrogram(
        linkage_matrix, color_threshold=dist)
    clss = scipy.cluster.hierarchy.fcluster(
        linkage_matrix, t=dist, criterion='distance')

    plt.show()

    return clss

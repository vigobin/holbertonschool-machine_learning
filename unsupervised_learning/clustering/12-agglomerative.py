#!/usr/bin/env python3
"""Agglomerative clustering"""

import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset:
    X is a numpy.ndarray of shape (n, d) containing the dataset.
    dist is the maximum cophenetic distance for all clusters.
    Performs agglomerative clustering with Ward linkage.
    Displays the dendrogram with each cluster displayed in a different color.
    Returns: clss, a numpy.ndarray of shape (n,) containing the cluster
        indices for each data point"""
    
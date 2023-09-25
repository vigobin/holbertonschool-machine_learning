#!/usr/bin/env python3
"""PCA function"""

import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset"""
    mean = np.mean(X, axis=0)
    centered_X = X - mean

    U, S, Vt = np.linalg.svd(centered_X, full_matrices=False)

    explained_variance = (S ** 2) / np.sum(S ** 2)

    cumulative_variance = np.cumsum(explained_variance)
    keep_components = np.argmax(cumulative_variance >= var) + 1

    transformation_matrix = Vt[:keep_components, :]

    # perform dimensionality reduction
    reduced_X = np.dot(centered_X, transformation_matrix.T)

    return reduced_X

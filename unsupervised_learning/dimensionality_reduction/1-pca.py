#!/usr/bin/env python3
"""PCA v2"""

import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset
    ndim: the new dimensionality of the transformed X.
    Returns: T, a numpy.ndarray of shape (n, ndim) containing the transformed
    version of X."""
    mean = np.mean(X, axis=0)
    centered_X = X - mean

    U, S, Vt = np.linalg.svd(centered_X, full_matrices=False)

    transformation_matrix = Vt[:ndim, :].T

    # perform dimensionality reduction
    return np.dot(centered_X, transformation_matrix)

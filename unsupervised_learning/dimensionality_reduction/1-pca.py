#!/usr/bin/env python3
"""PCA v2"""

import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset
    ndim: the new dimensionality of the transformed X.
    Returns: T, a numpy.ndarray of shape (n, ndim) containing the transformed
    version of X."""
    covariance_matrix = np.cov(X, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvectors[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    transformation_matrix = eigenvectors[:, :ndim]

    return np.dot(X, transformation_matrix)

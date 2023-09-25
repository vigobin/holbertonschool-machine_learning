#!/usr/bin/env python3
"""PCA v2"""

import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset
    ndim: the new dimensionality of the transformed X.
    Returns: T, a numpy.ndarray of shape (n, ndim) containing the transformed
    version of X."""
    
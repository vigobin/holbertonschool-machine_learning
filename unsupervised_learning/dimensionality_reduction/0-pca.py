#!/usr/bin/env python3
"""PCA function"""

import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset
    var is the fraction of the variance that the PCA transformation
        should maintain.
    Returns: the weights matrix, W, that maintains var fraction of
        X‘s original variance.
    W is a numpy.ndarray of shape (d, nd) where nd is the new
        dimensionality of the transformed X."""
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    explained_variance = (S ** 2) / np.sum(S ** 2)

    cumulative_variance = np.cumsum(explained_variance)
    keep_components = np.argmax(cumulative_variance >= var) + 1

    W = Vt[:keep_components + 1].T

    return W

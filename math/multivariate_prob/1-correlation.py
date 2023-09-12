#!/usr/bin/env python3
"""Correlation matrix"""

import numpy as np


def correlation(C):
    """Calculates a correlation matrix
        Returns a numpy.ndarray of shape (d, d) containing
        the correlation matrix."""
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    sd = np.sqrt(np.diag(C))
    # correlation matrix calculated from standard deviation along diagonal
    return C / np.outer(sd, sd)

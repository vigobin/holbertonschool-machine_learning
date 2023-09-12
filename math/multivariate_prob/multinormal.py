#!/usr/bin/env python3
"""Initialize class Multinormal"""

import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""
    def __init__(self, data):
        """Consructor"""
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

#!/usr/bin/env python3
"""Likelihood function"""

import numpy as np


def likelihood(x, n, P):
    """Calculates the likelihood of obtaining data given various
        hypothetical probabilities of developing severe side effects"""
    if n < 0:
        raise ValueError("n must be a positive integer")
    if x >= 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError('All values in P must be in the range [0, 1]')

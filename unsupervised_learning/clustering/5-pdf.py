#!/usr/bin/env python3
"""PDF function"""

import numpy as np


def pdf(X, m, S):
    """Calculates the probability density function of a Gaussian distribution.
    X is a numpy.ndarray of shape (n, d) containing the data points whose PDF
        should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of the distribution
    S is a numpy.ndarray of shape (d, d) containing the covariance of the
        distribution
    Returns: P, or None on failure
    P is a numpy.ndarray of shape (n,) containing the PDF values for
        each data point
    All values in P should have a minimum value of 1e-300"""

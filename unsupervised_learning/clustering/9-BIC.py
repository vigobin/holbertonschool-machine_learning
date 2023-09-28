#!/usr/bin/env python3
"""Bayesion Information Criterion"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using the
        Bayesian Information Criterion.
        Returns: best_k, best_result, li, bi, or None, None, None, None
            on failure.
        best_k is the best value for k based on its BIC
        best_result is tuple containing pi, m, S
        pi is a numpy.ndarray of shape (k,) containing the cluster priors.
        m is a numpy.ndarray of shape (k, d) containing the centroid means.
        S is a numpy.ndarray of shape (k, d, d) containing the
            covariance matrices.
        li is a numpy.ndarray of shape (kmax - kmin + 1) containing the
            log likelihood for each cluster size tested
        b is a numpy.ndarray of shape (kmax - kmin + 1) containing the
            BIC value for each cluster size tested
        Use: BIC = p * ln(n) - 2 * l
        p is the number of parameters required for the model
        n is the number of data points used to create the model
        li is the log likelihood of the model"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None, None, None
    if type(kmax) is not int or kmax <= 0:
        return None, None, None, None
    if kmax >= X.shape[0] or kmin >= X.shape[0]:
        return None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None
    if type(tol) is not float or tol <= 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None
    best_k = []
    best_result = []
    li = np.zeros(kmax - kmin + 1)
    bi = np.zeros(kmax - kmin + 1)

    return best_k, best_result, li, bi

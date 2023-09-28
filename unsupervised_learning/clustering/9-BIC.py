#!/usr/bin/env python3
"""Bayesion Information Criterion"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using the
        Bayesian Information Criterion.
        Returns: best_k, best_result, l, b, or None, None, None, None
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

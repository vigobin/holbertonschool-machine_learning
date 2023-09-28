#!/usr/bin/env python3
"""Expectation Maximization function"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs the expectation maximization for a GMM
    X is a numpy.ndarray of shape (n, d) containing the data set.
    k is a positive integer containing the number of clusters.
    tol is a non-negative float containing tolerance of the log likelihood,
        used to determine early stopping.
    If True, print Log Likelihood after {i} iterations:
        {l} every 10 iterations and after the last iteration
        {i} is the number of iterations of the EM algorithm
        {l} is the log likelihood, rounded to 5 decimal places
    Returns: pi, m, S, g, l, or None, None, None, None, None on failure
    pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
    m is a numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
        for each cluster
    g is a numpy.ndarray of shape (k, n) containing the probabilities for each
        data point in each cluster
    li is the log likelihood of the model"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None

    n, d = X.shape

    pi, m, S = initialize(X, k)

    prev_li = None

    for i in range(iterations):
        g, li = expectation(X, pi, m, S)

        pi, m, S = maximization(X, g)

        if prev_li is not None and abs(li - prev_li) <= tol:
            break

        prev_li = li

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, li.round(5)))

    if verbose:
        print("Log Likelihood after {} iterations: {}".format(
            i+1, li.round(5)))

    return pi, m, S, g, li

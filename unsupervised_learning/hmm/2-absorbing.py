#!/usr/bin/env python3
"""Absorbing Chains"""

import numpy as np


def absorbing(P):
    """Determines if a markov chain is absorbing.
    P is a is a square 2D numpy.ndarray of shape (n, n)
        representing the standard transition matrix.
    P[i, j] is the probability of transitioning from state i to state j.
    n is the number of states in the markov chain.
    Returns: True if it is absorbing, or False on failure."""
    if type(P) is not np.ndarray:
        return False

    n = P.shape[0]

    if P.shape != (n, n):
        return False

    Q = P[:-n, :-n]
    identity = np.eye(Q.shape[0])

    N = np.linalg.inv(np.eye(Q.shape[0]) - Q)

    abso = np.all(np.isclose(N[0], 0) | np.isfinite(N[0]))

    return abso

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

    if np.all(np.diag(P) == 1):
        return True

    Q = P[:-n, :-n]
    if Q.size == 0:
        return False

    identity = np.eye(Q.shape[0])
    N = np.linalg.inv(identity - Q)

    if not np.all(np.isfinite(N)):
        return False

    return np.all(np.all(np.isclose(N[0], 0) | np.isfinite(N)))

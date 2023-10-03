#!/usr/bin/env python3
"""Regular Chains"""

import numpy as np


def regular(P):
    """Determines the steady state probabilities of a regular markov chain
    P is a is a square 2D numpy.ndarray of shape (n, n) representing the
        transition matrix.
    P[i, j] is the probability of transitioning from state i to state j.
    n is the number of states in the markov chain.
    Returns: a numpy.ndarray of shape (1, n) containing the steady state
        probabilities, or None on failure."""
    if type(P) is not np.ndarray:
        return None

    n = P.shape[0]

    if P.shape != (n, n):
        return None

    eigvals, eigvecs = np.linalg.eig(P.T)

    steady_state_i = np.where(np.isclose(eigvals, 1))[0]

    if len(steady_state_i) != 1:
        return None

    steady_state_v = np.real(eigvecs[:, steady_state_i]).T

    steady_state_v /= np.sum(steady_state_v)

    return steady_state_v

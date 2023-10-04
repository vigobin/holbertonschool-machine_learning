#!/usr/bin/env python3
"""The Forward Algorithm"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Performs the forward algorithm for a hidden markov model:
    Observation is a numpy.ndarray of shape (T,) that contains the
        index of the observation.
        T is the number of observations.
    Emission is a numpy.ndarray of shape (N, M) containing the emission
        probability of a specific observation given a hidden state.
        Emission[i, j]: probability of observing j given the hidden state i
    N is the number of hidden states.
    M is the number of all possible observations.
    Transition is a 2D numpy.ndarray of shape (N, N) containing the
        transition probabilities.
    Transition[i, j] is the probability of transitioning from the hidden
        state i to j.
    Initial a numpy.ndarray of shape (N, 1) containing the probability of
        starting in a particular hidden state.
    Returns: P, F, or None, None on failure.
    P is the likelihood of the observations given the model
    F is a numpy.ndarray of shape (N, T) containing the
        forward path probabilities.
    F[i, j] is the probability of being in hidden state i at time j
        given the previous observations."""
    T = len(Observation)
    N, M = Emission.shape

    if T <= 0 or N <= 0 or M <= 0:
        return None, None

    F = np.zeros((N, T))

    F[:, 0] = Initial.squeeze() * Emission[:, Observation[0]]

    for t in range(1, T):
        for i in range(N):
            sum_term = np.sum(F[:, t - 1] * Transition[:, i])
            F[i, t] = Emission[i, Observation[t]] * sum_term

    P = np.sum(F[:, T - 1])

    return P, F

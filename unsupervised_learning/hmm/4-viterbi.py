#!/usr/bin/env python3
"""The Viretbi Algorithm"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Calculates the most likely sequence of hidden states for a
        hidden markov model.
    Observation is a numpy.ndarray of shape (T,) that contains the
        index of the observation.
    T is the number of observations.
    Emission is a numpy.ndarray of shape (N, M) containing the emission
        probability of a specific observation given a hidden state.
    Emission[i, j] is the probability of observing j given the hidden state i.
    N is the number of hidden states.
    M is the number of all possible observations.
    Transition is a 2D numpy.ndarray of shape (N, N) containing the
        transition probabilities.
    Transition[i, j] is the probability of transitioning from the
        hidden state i to j.
    Initial a numpy.ndarray of shape (N, 1) containing the probability of
        starting in a particular hidden state.
    Returns: path, P, or None, None on failure.
        path is the a list of length T containing the most likely
            equence of hidden states.
        P is the probability of obtaining the path sequence."""

    T = len(Observation)
    N, M = Emission.shape

    if T <= 0 or N <= 0 or M <= 0:
        return None, None

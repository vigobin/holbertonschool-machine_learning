#!/usr/bin/env python3
"""The Backward Algorithm"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Performs the backward algorithm for a hidden markov model.
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
    Returns: P, B, or None, None on failure.
        Pis the likelihood of the observations given the model.
        B is a numpy.ndarray of shape (N, T) containing the backward
            path probabilities.
        B[i, j] is the probability of generating the future observations
            from hidden state i at time j."""

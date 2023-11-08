#!/usr/bin/env python3
"""Bidirectional RNN"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Performs forward propagation for a bidirectional RNN:
        bi_cell is an instance of BidirectinalCell that will be used
            for the forward propagation.
        X is the data to be used, given as a numpy.ndarray of shape
            (t, m, i).
        t is the maximum number of time steps.
        m is the batch size.
        i is the dimensionality of the data.
        h_0 is the initial hidden state in the forward direction,
            given as a numpy.ndarray of shape (m, h).
        h is the dimensionality of the hidden state.
        h_t is the initial hidden state in the backward direction,
            given as a numpy.ndarray of shape (m, h).
        Returns: H, Y
        H is a numpy.ndarray containing all of the concatenated hidden states.
        Y is a numpy.ndarray containing all of the outputs."""
    t, m, i = X.shape
    _, h = h_0.shape
    Hf = np.zeros((t, m, h))
    Hb = np.zeros((t, m, h))
    Y = []

    h_prev = h_0
    h_next = h_t
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        h_next = bi_cell.backward(h_next, X[t-step-1])
        Hf[step] = h_prev
        Hb[t-step-1] = h_next

    H = np.concatenate((Hf, Hb), axis=1)
    Y = bi_cell.output(H)
    return H, Y

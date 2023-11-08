#!/usr/bin/env python3
"""Deep RNN"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN.
        rnn_cells is a list of RNNCell instances of length l that will be
            used for the forward propagation.
        l is the number of layers.
        X is the data to be used, given as a numpy.ndarray of shape (t, m, i).
        t is the maximum number of time steps.
        m is the batch size.
        i is the dimensionality of the data.
        h_0 is the initial hidden state, given as a numpy.ndarray of
            shape (l, m, h).
        h is the dimensionality of the hidden state.
        Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states.
        Y is a numpy.ndarray containing all of the outputs."""
    l, m, h = h_0.shape
    t, _, i = X.shape
    H = np.zeros((l, t, m, h))
    Y = []

    for i in range(l):
        h_prev = h_0[j]
        for j in range(t):
            x_t = X[j] if j == 0 else H[i-1, j]
            h_next, _, y = rnn_cells[j].forward(h_prev, h_prev, x_t)
            H[i, j] = h_next
            h_prev = h_next
        Y.append(y)
    
    return H, np.array(Y)

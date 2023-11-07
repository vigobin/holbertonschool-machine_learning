#!/usr/bin/env python3
"""LSTM Cell"""

import numpy as np


class LSTMCell:
    """Represents an LSTM unit"""

    def __init__(self, i, h, o):
        """i is the dimensionality of the data.
        h is the dimensionality of the hidden state.
        o is the dimensionality of the outputs.
        Creates the public instance attributes
            Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by
            that represent the weights and biases of the cell.
        Wfand bf are for the forget gate.
        Wuand bu are for the update gate.
        Wcand bc are for the intermediate cell state.
        Woand bo are for the output gate.
        Wyand by are for the outputs.
        The weights should be initialized using a random normal distribution
            in the order listed above.
        The weights will be used on the right side for matrix multiplication.
        The biases should be initialized as zeros."""

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one time step
        x_t is a numpy.ndarray of shape (m, i) that contains
            the data input for the cell.
        m is the batche size for the data.
        h_prev is a numpy.ndarray of shape (m, h) containing the
            previous hidden state.
        c_prev is a numpy.ndarray of shape (m, h) containing the
            previous cell state.
        The output of the cell should use a softmax activation function.
        Returns: h_next, c_next, y
        h_next is the next hidden state.
        c_next is the next cell state.
        y is the output of the cell."""

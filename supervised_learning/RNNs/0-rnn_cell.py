#!/usr/bin/env python3
"""RNN Cell"""

import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN"""
    def __init__(self, i, h, o):
        """Creates the public instance attributes:
        i is the dimensionality of the data.
        h is the dimensionality of the hidden state.
        o is the dimensionality of the outputs.
        Wh, Wy, bh, by represent the weights and biases of the cell:
        Wh and bh are for the concatenated hidden state and input data.
        Wy and by are for the output."""
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, 0))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step:
        x_t is a numpy.ndarray of shape (m, i) that contains the data
            input for the cell.
        m is the batche size for the data.
        h_prev is a numpy.ndarray of shape (m, h) containing the
            previous hidden state.
        The output of the cell should use a
            softmax activation function.
        Returns: h_next, y
        h_next is the next hidden state.
        y is the output of the cell."""
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(h_x, self.Wh) + self.bh)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, y

    def softmax(self, x):
        """softmax function"""
        output = np.exp(x - np.max(x))
        return output / output.sum(axis=1, keepdims=True)

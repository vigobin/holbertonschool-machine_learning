#!/usr/bin/env python3
"""GRU Cell"""

import numpy as np


class GRUCell:
    """Represents a gated recurrent unit"""
    def __init__(self, i, h, o):
        """i is the dimensionality of the data.
    h is the dimensionality of the hidden state.
    o is the dimensionality of the outputs.
    Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
        that represent the weights and biases of the cell.
    Wzand bz are for the update gate.
    Wrand br are for the reset gate.
    Whand bh are for the intermediate hidden state.
    Wyand by are for the output.
    The weights should be initialized using a random normal distribution in
        the order listed above.
    The weights will be used on the right side for matrix multiplication.
    The biases should be initialized as zeros."""
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step.
        x_t is a numpy.ndarray of shape (m, i) that contains the data input
            for the cell.
        m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
            hidden state.
        The output of the cell should use a softmax activation function.
        Returns: h_next, y
        h_next is the next hidden state.
        y is the output of the cell."""
        input = np.concatenate((h_prev, x_t), axis=1)
        update = np.dot(input, self.Wz) + self.bz
        reset = np.dot(input, self.Wr) + self.br
        hidden = np.tanh(np.dot(np.concatenate((
            reset * h_prev, x_t), axis=1), self.Wh) + self.bh)
        h_next = (1 - update) * h_prev + update * hidden
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y

    def softmax(self, x):
        """Softmax function"""
        output = np.exp(x - np.max(x))
        return output / output.sum(axis=1, keepdims=True)

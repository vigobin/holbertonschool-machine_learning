#!/usr/bin/env python3
"""GRU Cell"""

import numpy as np


class GRUCell:
    """Represents a gated recurrent unit"""
    def __init__(self, i, h, o):
        """i is the dimensionality of the data
    h is the dimensionality of the hidden state
    o is the dimensionality of the outputs
    Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by that represent the weights and biases of the cell
    Wzand bz are for the update gate
    Wrand br are for the reset gate
    Whand bh are for the intermediate hidden state
    Wyand by are for the output
    The weights should be initialized using a random normal distribution in the order listed above
    The weights will be used on the right side for matrix multiplication
    The biases should be initialized as zeros."""

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step
        x_t is a numpy.ndarray of shape (m, i) that contains the data input for the cell
        m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous hidden state
        The output of the cell should use a softmax activation function
        Returns: h_next, y
        h_next is the next hidden state
        y is the output of the cell."""

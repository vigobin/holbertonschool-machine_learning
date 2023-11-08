#!/usr/bin/env python3
"""Bidirectional Cell Forward"""

import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of an RNN."""

    def __init__(self, i, h, o):
        """i is the dimensionality of the data.
        h is the dimensionality of the hidden states.
        o is the dimensionality of the outputs.
        Creates the public instance attributes Whf, Whb, Wy, bhf, bhb, by
            that represent the weights and biases of the cell.
        Whf and bhfare for the hidden states in the forward direction.
        Whb and bhbare for the hidden states in the backward direction.
        Wy and byare for the outputs.
        The weights should be initialized using a random normal distribution
            in the order listed above.
        The weights will be used on the right side for matrix multiplication.
        The biases should be initialized as zeros."""

    def forward(self, h_prev, x_t):
        """Calculates the hidden state in the forward direction
            for one time step.
            x_t is a numpy.ndarray of shape (m, i) that contains the data
                input for the cell.
            m is the batch size for the data.
            h_prev is a numpy.ndarray of shape (m, h) containing the previous
                hidden state.
            Returns: h_next, the next hidden state."""

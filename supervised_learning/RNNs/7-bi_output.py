#!/usr/bin/env python3
"""Bidirectional Output"""

import numpy as np


def output(self, H):
    """Calculates all outputs for the RNN:
        H is a numpy.ndarray of shape (t, m, 2 * h) that contains the
            concatenated hidden states from both directions,
            excluding their initialized states.
        t is the number of time steps.
        m is the batch size for the data.
        h is the dimensionality of the hidden states.
        Returns: Y, the outputs."""

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
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(i + h, h))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, h))

    def softmax(self, x):
        """Softmax function"""
        output = np.exp(x - np.max(x))
        return output / output.sum(axis=1, keepdims=True)

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
        input = np.concatenate((h_prev, x_t), axis=1)
        forget_gate = np.dot(input, self.Wf) + self.bf
        update = np.dot(input, self.Wu) + self.bu
        cell_state = np.tanh(np.dot(input, self.Wc) + self.bc)
        cell_next = forget_gate * c_prev + update * cell_state
        output_gate = np.dot(input, self.Wo) + self.bo
        h_next = output_gate * np.tanh(cell_next)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        
        return h_next, cell_next, y

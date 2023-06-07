#!/usr/bin/env python3
"""Deep Neural Network"""

import numpy as np


class DeepNeuralNetwork:
    """defines a deep neural network with one hidden layer
    performing binary classification"""

    def __init__(self, nx, layers):
        """Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        weights = {}
        previous = nx
        for i, layer in enumerate(layers, 1):
            if type(layer) is not int or layer < 0:
                raise TypeError('layers must be a list of positive integers')
            weights['b{}'.format(i)] = np.zeros((layer, 1))
            weights['W{}'.format(i)] = np.random.randn(
                layer, previous) * np.sqrt(2 / previous)
            previous = layer

        self.L = len(layers)
        self.cache = {}
        self.weights = weights

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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        previous_node = nx
        for i, layer in enumerate(layers, 1):
            if type(layer) is not int or layer < 0:
                raise TypeError('layers must be a list of positive integers')
            self.__weights['b{}'.format(i)] = np.zeros((layer, 1))
            self.__weights['W{}'.format(i)] = np.random.randn(
                layer, previous_node) * np.sqrt(2 / previous_node)
            previous_node = layer

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

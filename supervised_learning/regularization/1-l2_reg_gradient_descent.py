#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network using
    gradient descent with L2 regularization"""
    m = Y.shape[1]

    for layer in reversed(range(1, L + 1)):
        A = cache['A' + str(layer)]
        A_prev = cache['A' + str(layer - 1)]
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]
        # dZ = None

        if layer == L:
            dZ = A - Y
        else:
            dA = np.dot(weights['W' + str(layer + 1)].T, dZ)
            dZ = dA * (1 - np.power(A, 2))

        dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db

    return weights

#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network using
    gradient descent with L2 regularization"""
    m = Y.shape[1]
    for layer in range(L, 0, -1):
        A_cur = cache["A{}".format(layer)]
        A_prev = cache["A{}".format(layer - 1)]
        if layer == L:
            dz = (A_cur - Y)
        else:
            dz = dAPrevLayer * (1 - np.square(A_cur))
        W = weights["W{}".format(layer)]
        l2 = (lambtha/m) * W
        dW = np.matmul(dz, A_prev.T)/Y.shape[1] + l2
        db = np.sum(dz, axis=1, keepdims=True)/Y.shape[1]
        dAPrevLayer = np.matmul(W.T, dz)
        weights["W{}".format(layer)] -= alpha * dW
        weights["b{}".format(layer)] -= alpha * db

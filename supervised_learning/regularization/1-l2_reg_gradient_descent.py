#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network using
    gradient descent with L2 regularization"""
    m = Y.shape[1]
    back = {}
    for index in range(L, 0, -1):
        A = cache["A{}".format(index - 1)]
        if index == L:
            back["dz{}".format(index)] = (cache["A{}".format(index)] - Y)
        else:
            dz_prev = back["dz{}".format(index + 1)]
            A_current = cache["A{}".format(index)]
            back["dz{}".format(index)] = (
                np.matmul(W_prev.transpose(), dz_prev) *
                (A_current * (1 - A_current)))
        dz = back["dz{}".format(index)]
        dW = (1 / m) * (
            (np.matmul(dz, A.transpose())) + (
                lambtha * weights["W{}".format(index)]))
        db = (1 / m) * (
            (np.sum(dz, axis=1, keepdims=True)) + (
                lambtha * weights["b{}".format(index)]))
        W_prev = weights["W{}".format(index)]
        weights["W{}".format(index)] = (
            weights["W{}".format(index)] - (alpha * dW))
        weights["b{}".format(index)] = (
            weights["b{}".format(index)] - (alpha * db))

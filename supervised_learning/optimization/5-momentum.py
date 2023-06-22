#!/usr/bin/env python3
"""Momentum function"""

import tensorflow as tf


def update_variables_momentum(alpha, beta1, var, grad, v):
    """updates a variable using the gradient descent
    with momentum optimization algorithm"""
    dW_prev = (beta1 * v) + ((1 - beta1) * grad)
    var -= alpha * dW_prev
    return var, dW_prev

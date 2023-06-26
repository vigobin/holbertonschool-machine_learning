#!/usr/bin/env python3
"""Adam optimization"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Updates a variable in place using the Adam optimization algorithm"""
    v_dW = (beta1 * v) + ((1 - beta1) * grad)
    s_dW = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    v_dW_c = v_dW / (1 - (beta1 ** t))
    s_dW_c = s_dW / (1 - (beta2 ** t))
    var -= alpha * (v_dW_c / (epsilon + (s_dW_c ** (1 / 2))))
    return var, v_dW, s_dW

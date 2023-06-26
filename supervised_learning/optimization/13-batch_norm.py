#!/usr/bin/env python3
"""Batch Normalization"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output of a neural network using
    batch normalization"""
    Z_norm = Z.copy()
    m_mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Z_norm = (Z - m_mean)/(var + epsilon) ** 0.5
    return (Z_norm*gamma) + beta

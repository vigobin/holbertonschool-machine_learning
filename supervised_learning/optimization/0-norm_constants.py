#!/usr/bin/env python3
"""Normalization Constants"""

import numpy as np


def normalization_constants(X):
    """Calculates the normalization (standardization) constants of a matrix"""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return mean, std

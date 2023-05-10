#!/usr/bin/env python3
"""Cat's Got Your Tongue """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Return a new numpy array with the concatenation of two matrices"""
    return np.concatenate((mat1, mat2), axis=axis)

#!/usr/bin/env python3
"""Definiteness of matrix"""

import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix
        Returns: The string Positive definite, Positive semi-definite,
            Negative semi-definite, Negative definite, or Indefinite"""
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.shape[0] != matrix.shape[1]:
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if np.all(eigenvalues) > 0:
        return "Positive definite"
    if np.all(eigenvalues) >= 0:
        return "Positive semi-definite"
    if np.all(eigenvalues) < 0:
        return "Negative definite"
    if np.all(eigenvalues) <= 0:
        return "Negative semi-definite"
    else:
        return "Indefinite"

#!/usr/bin/env python3
"""Function that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """Returns list of dimensions of matrix"""
    matrix_dim = []
    while (type(matrix) is list):
        matrix_dim.append(len(matrix))
        matrix = matrix[0]
    return matrix_dim

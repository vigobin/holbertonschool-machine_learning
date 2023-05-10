#!/usr/bin/env python3
""" that concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Returns a new matrix of concatenated matrices"""
    if axis == 0:
        return mat1 + mat2
    elif axis == 1:
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None

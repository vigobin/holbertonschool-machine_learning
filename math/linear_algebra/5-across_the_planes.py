#!/usr/bin/env python3
"""Function that adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Returns a new matrix with the sum"""
    if len(mat1) != len(mat2):
        return None
    sum_list = []
    for i, row in enumerate(mat1):
        sum_list.append([])
        for j in range(len(row)):
            sum_list[i].append(mat1[i][j] + mat2[i][j])
    return sum_list

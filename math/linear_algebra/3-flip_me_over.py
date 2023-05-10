#!/usr/bin/env python3
"""Function to transpose a 2D matrix"""


def matrix_transpose(matrix):
    """Returns a matrix which transposes a 2d one"""
    new_matrix = []
    for i, row in enumerate(matrix):
        if i == 0:
            for i in row:
                new_matrix.append([])
        for index, j in enumerate(row):
            new_matrix[index].append(j)
    return new_matrix

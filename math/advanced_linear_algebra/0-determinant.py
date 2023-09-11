#!/usr/bin/env python3
"""Determinant of Matrix"""


def determinant(matrix):
    """Calculates the determinant of a matrix"""
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    for i in matrix:
        if type(i) is not list:
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    if n == 0:
        return 1

    if n == 1:
        return matrix[0][0]

    if n == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return a * d - b * c

    det = 0
    for j in range(n):
        sub_matrix = [row[:j] + row[j + 1:] for row in matrix[1:]]
        cofactor = matrix[0][j] * determinant(sub_matrix)
        #  Add or subtract cofactor to the determinant with alternating signs
        det += cofactor if j % 2 == 0 else -cofactor

    return det

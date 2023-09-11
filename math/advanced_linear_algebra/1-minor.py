#!/usr/bin/env python3
"""Minor of a matrix"""


def minor(matrix):
    """Calculates the minor matrix of a matrix"""
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    for i in matrix:
        if type(i) is not list:
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = []

    for i in range(n):
        minor_row = []
        for j in range(n):
            # Calculate the minor by removing the i-th row and j-th column
            #   and finding the determinant.
            sub_matrix = [row[:j] + row[j + 1:] for row in (
                matrix[:i] + matrix[i + 1:])]
            minor_value = determinant(sub_matrix)
            minor_row.append(minor_value)
        minor_matrix.append(minor_row)

    return minor_matrix


def determinant(matrix):
    """Calculates the determinant of a matrix"""
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    for i in matrix:
        if type(i) is not list:
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    if n == 0:
        return 1

    if n == 1 and len(matrix[0]) == 0:
        return 1

    if n == 1:
        return matrix[0][0]

    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

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

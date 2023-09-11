#!/usr/bin/env python3
"""Inverse Matrix"""


def inverse(matrix):
    """Calculates the inverse of a matrix
        Returns: the inverse of matrix, or None if matrix is singular"""
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    for i in matrix:
        if type(i) is not list:
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    det_mat = determinant(matrix)

    if det_mat == 0:
        return None

    cofactors = cofactor(matrix)
    adjugate_mat = [[cofactors[j][i] for j in range(n)] for i in range(n)]
    # Calculate inverse by multiplying the adjugate matrix by the
    #   reciprocal of the determinant.
    inverse_matrix = [[1 / det_mat * adjugate_mat[i][j] for j in range(
        n)for i in range(n)]]

    return inverse_matrix


def adjugate(matrix):
    """Calculates the adjugate matrix of a matrix"""

    cofactors = cofactor(matrix)
    adjugate_matrix = [[cofactors[j][i] for j in range(len(cofactors))]
                       for i in range(len(cofactors))]
    return adjugate_matrix


def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix"""
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    for i in matrix:
        if type(i) is not list:
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return[[1]]

    cofactor_matrix = []

    for i in range(n):
        cofactor_row = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j + 1:] for row in (
                matrix[:i] + matrix[i + 1:])]
            minor_value = determinant(sub_matrix)
            sign = (-1) ** (i + j)
            cofactor_value = minor_value * sign
            cofactor_row.append(cofactor_value)

        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix


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

    if n == 1:
        return[[1]]

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

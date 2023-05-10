#!/usr/bin/env python3
"""Bracing The Elements """


def np_elementwise(mat1, mat2):
    """Function that performs element-wise addition, subtraction,
    multiplication, and division"""
    results = []
    results.append(mat1 + mat2)
    results.append(mat1 - mat2)
    results.append(mat1 * mat2)
    results.append(mat1 / mat2)
    return tuple(results)

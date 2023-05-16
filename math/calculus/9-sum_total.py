#!/usr/bin/env python3
"""Calculates sum total"""


def summation_i_squared(n):
    """Returns the integer value of the sum"""
    if type(n) is not int or n < 1:
        return None

    return n + summation_i_squared(n - 1)

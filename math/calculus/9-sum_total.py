#!/usr/bin/env python3
"""Calculates sum total"""


def summation_i_squared(n):
    """Returns the integer value of the sum"""
    if type(n) is not int or n < 1:
        return None
    elif n == 1:
        return 1
    else:
        return n**2 + summation_i_squared(n - 1)

#!/usr/bin/env python3
"""Calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """Returns a list of coefficients representing the
    derivative of the polynomial"""
    if type(poly) is not list or len(poly) < 1:
        return None

    derivative = []
    for i in range(1, len(poly)):
        coef = poly[i] * i
        derivative.append(coef)

    if len(derivative) == 0:
        return [0]

    return derivative

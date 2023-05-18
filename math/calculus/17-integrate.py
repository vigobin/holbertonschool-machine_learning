#!/usr/bin/env python3
"""Calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Return a new list of coefficients representing
    the integral of the polynomial"""
    if type(poly) is not list or len(poly) < 1:
        return None
    if type(C) is not int:
        return None

    integral = [C]
    for i in range(len(poly)):
        coef = poly[i] / (i+1)
        if coef.is_integer():
            integral.append(int(coef))
        else:
            integral.append(coef)

    """ while len(integral) > 1 and integral[-1] == 0:
        integral.pop() """

    return integral

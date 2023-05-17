#!/usr/bin/env python3
"""Calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Return a new list of coefficients representing
    the integral of the polynomial"""
    if type(poly) is not list:
        return None
    if type(C) is not int:
        return None

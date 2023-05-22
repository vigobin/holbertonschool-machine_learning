#!/usr/bin/env python3
"""Represents an exponential distribution"""


class Exponential:
    """Defines an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Class contructor"""
        if data is None:
            if lambtha < 1:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                lambtha = float(len(data) / sum(data))
                self.lambtha = lambtha

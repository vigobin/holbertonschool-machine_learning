#!/usr/bin/env python3
"""Represents a normal distribution"""


class Normal:
    """Defins the Normal class"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Class contructor """
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.mean = sum(data) / len(data)
                variance = sum((x - mean) ** 2 for x in data) / len(data)
                self.stddev = variance ** 0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return self.mean + z * self.stddev

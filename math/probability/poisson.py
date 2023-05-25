#!/usr/bin/env python3
"""Represents a poisson distribution"""


class Poisson:
    """class Poisson that represents a poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Class constructor"""
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
                lambtha = float(sum(data) / len(data))
                self.lambtha = lambtha

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        factorial = 1
        for i in range(1, k + 1):
            factorial *= self.lambtha / i
        pmf = factorial * pow(2.7182818285, -self.lambtha)
        return pmf

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)

        return cdf

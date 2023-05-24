#!/usr/bin/env python3
"""Represents a binomial distribution"""


class Binomial:
    """Defines the class Binomial"""

    def __init__(self, data=None, n=1, p=0.5):
        """Class constructor"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not 0 < p < 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(round(n))
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.n, self.p = self.calculate_n_p(data)

    def calculate_n_p(self, data):
        """"""
        p = sum(data) / len(data)
        n = int(round(p * len(data)))
        p = p * n / len(data)
        return n, p

    def pmf(self, k):
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        coefficient = self._binomial_coefficient(self.n, k)
        probability = self.p ** k * (1 - self.p) ** (self.n - k)
        return coefficient * probability

    def _binomial_coefficient(self, n, k):
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result


#!/usr/bin/env python3
"""Initialize class Multinormal"""

import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""
    def __init__(self, data):
        """Consructor"""
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        self.cov = np.zeros((d, d))

        for i in range(d):
            for j in range(d):
                self.cov[i, j] = np.sum((data[i, :] - self.mean[i, :]) * (
                    data[j, :] - self.mean[j, :])) / (n - 1)

    def pdf(self, x):
        """Calculates the PDF at a data point"""
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({d}, 1)")

        diff = x - self.mean
        exponent = -0.5 * np.dot(np.dot(diff.T, np.linalg.inv(self.cov)), diff)
        coef = 1 / np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov))
        pdf_value = coef * np.exp(exponent)

        return pdf_value

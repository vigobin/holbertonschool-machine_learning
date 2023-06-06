#!/usr/bin/env python3
"""Neuron class"""

import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """Defines a single neuron performing binary classification"""
    def __init__(self, nx):
        """Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0
        self.__costs = []

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        fp = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-fp))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(
            1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dZ = A - Y
        dW = (1 / m) * np.matmul(X, dZ.T)
        db = (1 / m) * np.sum(dZ)

        self.__W = self.__W - alpha * dW.T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the neuron by updating the private attributes
        __W, __b, and __A"""
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if step <= 0 or step > iterations:
            raise ValueError('step must be positive and <= iterations')

        for i in range(iterations + 1):
            A = self.forward_prop(X)
            cost = self.cost(Y, A)
            self.gradient_descent(X, Y, A, alpha)
            self.__costs.append(cost)

            if verbose and i % step == 0:
                print(f'Cost after {i} iterations: {cost}')

        if graph or verbose:
            x = np.arange(0, iterations + 1, step)
            y = self.__costs[::step]

            if verbose:
                print('Cost after {} iterations: {}'.format(
                    iterations, self.__costs[-1]))

            if graph:
                plt.plot(x, y, 'b-')
                plt.xlabel('Iteration')
                plt.ylabel('Cost')
                plt.title('Training Cost')
                plt.show()

        return self.evaluate(X, Y)

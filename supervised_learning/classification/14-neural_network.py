#!/usr/bin/env python3
"""Neural Network"""

import numpy as np


class NeuralNetwork:
    """defines a neural network with one hidden layer
    performing binary classification"""

    def __init__(self, nx, nodes):
        """Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        fp1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-fp1))
        fp2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-fp2))
        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(
            1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions"""
        A1, A2 = self.forward_prop(X)
        prediction = np.where(A2 >= 0.5, 1, 0)
        cost = self.cost(Y, A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the nueral network"""
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(dZ2, A1.transpose())
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.matmul(self.W2.transpose(), dZ2) * (A1 * (1 - A1))
        dW1 = (1 / m) * np.matmul(dZ1, X.transpose())
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        self.__W2 = self.W2 - (alpha * dW2)
        self.__b2 = self.b2 - (alpha * db2)
        self.__W1 = self.W1 - (alpha * dW1)
        self.__b1 = self.__b1 - (alpha * db1)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network"""
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)

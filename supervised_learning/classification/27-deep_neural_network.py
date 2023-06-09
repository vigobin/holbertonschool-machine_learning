#!/usr/bin/env python3
"""Deep Neural Network"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """defines a deep neural network with one hidden layer
    performing binary classification"""

    def __init__(self, nx, layers):
        """Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        previous_node = nx
        for i, layer in enumerate(layers, 1):
            if type(layer) is not int or layer < 0:
                raise TypeError('layers must be a list of positive integers')
            self.__weights['b{}'.format(i)] = np.zeros((layer, 1))
            self.__weights['W{}'.format(i)] = np.random.randn(
                layer, previous_node) * np.sqrt(2 / previous_node)
            previous_node = layer

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            W = self.__weights['W{}'.format(i)]
            b = self.__weights['b{}'.format(i)]
            A_prev = self.__cache['A{}'.format(i - 1)]
            Z = np.matmul(W, A_prev) + b
            if i == self.__L:
                A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
            else:
                A = 1 / (1 + np.exp(-Z))
            self.__cache['A{}'.format(i)] = A

        return self.__cache['A{}'.format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return (prediction, cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        L = self.__L

        dZ = cache['A' + str(L)] - Y
        for i in range(L, 0, -1):
            A_prev = cache['A' + str(i-1)]
            W_key = 'W' + str(i)
            b_key = 'b' + str(i)

            dW = (1/m) * np.matmul(dZ, A_prev.T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.matmul(self.__weights[W_key].T, dZ)

            self.__weights[W_key] -= alpha * dW
            self.__weights[b_key] -= alpha * db

            if i != 1:
                dZ = dA_prev * (A_prev * (1 - A_prev))

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the deep neural network"""
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')

        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        # Check verbose and graph parameters
        if (verbose or graph):
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        # Initialize cost list for graphing
        costs = []

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

            # Calculate and store cost every step iterations
            if i % step == 0 or i == iterations - 1:
                cost = self.cost(Y, A)
                costs.append(cost)

                # Print cost if verbose is True
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))

        # Plot the training data if graph is True
        if graph:
            plt.plot(range(0, iterations, step) + [iterations - 1],
                     costs, 'b-')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('Training Cost')
            plt.show()

        # Return evaluation of the training data
        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith('.pkl'):
            filename = filename[:] + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        f.close()

    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except FileNotFoundError:
            return None

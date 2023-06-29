#!/usr/bin/env python3
"""Forward Propagation with Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout"""
    cache = {}
    cache["A0"] = X
    for lay in range(1, L + 1):
        W = weights["W{}".format(lay)]
        A = cache["A{}".format(lay - 1)]
        b = weights["b{}".format(lay)]
        z = np.matmul(W, A) + b

        if lay == L:
            softA = np.exp(z)/np.sum(np.exp(z), axis=0)
            cache["A{}".format(lay)] = softA

        else:
            newA = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
            d = (np.random.rand(newA.shape[0], newA.shape[1]) < keep_prob)
            cache["D{}".format(lay)] = d.astype(int)
            newA *= d
            newA /= keep_prob
            cache["A{}".format(lay)] = newA

    return cache

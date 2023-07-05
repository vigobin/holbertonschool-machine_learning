#!/usr/bin/env python3
"""13 Predict"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Makes a prediction using a neural network"""
    prediction = network.predict(data, verbose=verbose)
    return prediction

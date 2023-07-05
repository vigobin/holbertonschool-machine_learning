#!/usr/bin/env python3
"""12 Test"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """that tests a neural network"""
    loss, accuracy = network.evaluate(data, labels, verbose=verbose)
    return loss, accuracy

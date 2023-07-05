#!/usr/bin/env python3
"""Save and Load Configuration"""

import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model's configuration in JSON format"""
    config = network.to_json()
    with open(filename, 'w') as file:
        file.write(config)


def load_config(filename):
    """Loads a model with a specific configuration"""
    with open(filename, 'r') as file:
        config = file.read()
    model = K.models.model_from_json(config)
    return model

#!/usr/bin/env python3
"""Transition Layer"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in
    Densely Connected Convolutional Networks

    X: Output from the previous layer
    nb_filters: Number of filters in X
    compression: Compression factor for the transition layer

    Returns: Output of the transition layer and the number of filters
    within the output, respectively
    """

    # Calculate the number of output filters
    nb_filters = int(nb_filters * compression)

    # Batch normalization
    X = K.layers.BatchNormalization()(X)

    # ReLU activation
    X = K.layers.Activation('relu')(X)

    # Convolution layer
    X = K.layers.Conv2D(
        filters=nb_filters, kernel_size=(1, 1), padding='same',
        kernel_initializer='he_normal'
    )(X)

    # Average pooling layer
    X = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
                                  padding='valid')(X)

    return X, nb_filters

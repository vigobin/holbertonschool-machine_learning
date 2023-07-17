#!/usr/bin/env python3
"""Projection Block"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block as described in
        Deep Residual Learning for Image Recognition (2015):

    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution as well as
        the 1x1 convolution in the shortcut connection
    s is the stride of the first convolution in both the main path
    and the shortcut connection
    All convolutions inside the block should be followed by batch normalization
    along the channels axis and a rectified linear activation (ReLU),
    respectively.
    All weights should use he normal initialization
    Returns: the activated output of the projection block
"""
    F11, F3, F12 = filters

    X_shortcut = A_prev

    # First component of main path
    X = K.layers.Conv2D(F11, (1, 1), strides=(s, s), padding='same',
                        kernel_initializer='he_normal')(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component of main path
    X = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component of main path
    X = K.layers.Conv2D(F12, (1, 1), padding='same',
                        kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut path
    X_shortcut = K.layers.Conv2D(
        F12, (1, 1), strides=(s, s), padding='same',
        kernel_initializer='he_normal')(X_shortcut)
    X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)

    # Add shortcut value to main path
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X

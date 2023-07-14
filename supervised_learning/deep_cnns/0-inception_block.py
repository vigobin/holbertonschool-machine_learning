#!/usr/bin/env python3
"""Inception Block"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in
    Going Deeper with Convolutions(2014).

    A_prev: Output from the previous layer
    filters: Tuple or list containing F1, F3R, F3, F5R, F5, FPP, respectively:
    F1: Number of filters in the 1x1 convolution
    F3R: Number of filters in the 1x1 convolution before the 3x3 convolution
    F3: Number of filters in the 3x3 convolution
    F5R: Number of filters in the 1x1 convolution before the 5x5 convolution
    F5: Number of filters in the 5x5 convolution
    FPP: Number of filters in the 1x1 convolution after the max pooling

    Returns: Concatenated output of the inception block
    """

    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution
    conv1 = K.layers.Conv2D(
        filters=F1, kernel_size=(1, 1), padding='same', activation='relu'
    )(A_prev)

    # 1x1 convolution before the 3x3 convolution
    conv3R = K.layers.Conv2D(
        filters=F3R, kernel_size=(1, 1), padding='same', activation='relu'
    )(A_prev)

    # 3x3 convolution
    conv3 = K.layers.Conv2D(
        filters=F3, kernel_size=(3, 3), padding='same', activation='relu'
    )(conv3R)

    # 1x1 convolution before the 5x5 convolution
    conv5R = K.layers.Conv2D(
        filters=F5R, kernel_size=(1, 1), padding='same', activation='relu'
    )(A_prev)

    # 5x5 convolution
    conv5 = K.layers.Conv2D(
        filters=F5, kernel_size=(5, 5), padding='same', activation='relu'
    )(conv5R)

    # Max pooling
    pool = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(1, 1), padding='same'
    )(A_prev)

    # 1x1 convolution after max pooling
    convPP = K.layers.Conv2D(
        filters=FPP, kernel_size=(1, 1), padding='same', activation='relu'
    )(pool)

    # Concatenate all the branches
    concat = K.layers.concatenate([conv1, conv3, conv5, convPP], axis=-1)

    return concat

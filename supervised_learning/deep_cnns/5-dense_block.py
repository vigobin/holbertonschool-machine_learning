#!/usr/bin/env python3
"""Dense Block """

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in
    Densely Connected Convolutional Networks

    X: Output from the previous layer
    nb_filters: Number of filters in X
    growth_rate: Growth rate for the dense block
    layers: Number of layers in the dense block

    Returns: Concatenated output of each layer within the dense block and the
    number of filters within the concatenated outputs
    """

    concat = X
    filters = nb_filters

    for _ in range(layers):
        # 1x1 bottleneck convolution
        bottleneck = K.layers.BatchNormalization()(concat)
        bottleneck = K.layers.Activation('relu')(bottleneck)
        bottleneck = K.layers.Conv2D(
            filters=4 * growth_rate, kernel_size=(1, 1), padding='same',
            kernel_initializer='he_normal'
        )(bottleneck)

        # 3x3 convolution
        conv = K.layers.BatchNormalization()(bottleneck)
        conv = K.layers.Activation('relu')(conv)
        conv = K.layers.Conv2D(
            filters=growth_rate, kernel_size=(3, 3), padding='same',
            kernel_initializer='he_normal'
        )(conv)

        # Concatenate the output with the previous layers
        concat = K.layers.concatenate([concat, conv], axis=-1)

        # Update the number of filters
        filters += growth_rate

    return concat, filters

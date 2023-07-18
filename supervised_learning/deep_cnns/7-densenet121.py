#!/usr/bin/env python3
"""Task 7 -  DenseNet-121"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    input_shape = (224, 224, 3)

    inputs = K.Input(shape=input_shape)
    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=K.regularizers.l2(1e-4))(inputs)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='same')(X)

    nb_filters = 64

    # Dense Block 1
    X, nb_filters = dense_block(X, 6, nb_filters, growth_rate)
    # Transition Layer 1
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2
    X, nb_filters = dense_block(X, 12, nb_filters, growth_rate)
    # Transition Layer 2
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3
    X, nb_filters = dense_block(X, 24, nb_filters, growth_rate)
    # Transition Layer 3
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4
    X, nb_filters = dense_block(X, 16, nb_filters, growth_rate)

    X = K.layers.GlobalAveragePooling2D()(X)
    outputs = K.layers.Dense(1000, activation='softmax')(X)

    model = K.Model(inputs=inputs, outputs=outputs)
    return model

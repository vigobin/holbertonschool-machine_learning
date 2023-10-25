#!/usr/bin/env python3
"""Convolutional Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder:
    input_dims is an integer containing the dimensions of the model input.
    filters is a list containing the number of filters for each convolutional
        layer in the encoder, respectively.
            The filters should be reversed for the decoder.
    latent_dims is a tuple of integers containing the dimensions of the
        latent space representation.
    Each convolution in the encoder should use a kernel size of (3, 3) with
        same padding and relu activation, followed by max pooling size (2, 2).
    Each convolution in the decoder, except for the last two, should use a
        filter size of (3, 3) with same padding and relu activation,
        followed by upsampling of size (2, 2).
    The second to last convolution should instead use valid padding.
    The last convolution should have the same number of filters as the number
        of channels in input_dims with sigmoid activation and no upsampling.
    Returns: encoder, decoder, auto
    encoder is the encoder model.
    decoder is the decoder model.
    auto is the full autoencoder model.
    The autoencoder model should be compiled using adam optimization and
        binary cross-entropy loss."""
    # Encoder model
    encoder_inputs = keras.layers.Input(shape=input_dims)
    x = encoder_inputs
    for num_filters in range(len(filters)):
        x = keras.layers.Conv2D(filters[num_filters], (3, 3),
                                activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    encoder = keras.Model(encoder_inputs, x)

    # Decoder model
    decoder_inputs = keras.layers.Input(shape=latent_dims)
    y = decoder_inputs
    padding = 'same'
    for num_filters in range(len(filters) - 1, -1, -1):
        if num_filters == 0:
            padding = 'valid'

        y = keras.layers.Conv2D(filters[num_filters], (3, 3),
                                activation='relu', padding=padding)(y)
        y = keras.layers.UpSampling2D((2, 2))(y)

    decoder_outputs = keras.layers.Conv2D(
        input_dims[-1], (3, 3), activation='sigmoid', padding='same')(y)

    decoder = keras.Model(decoder_inputs, decoder_outputs)

    # Autoencoder model
    auto = keras.Model(
        encoder_inputs, decoder(encoder(encoder_inputs)))

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

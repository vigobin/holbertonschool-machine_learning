#!/usr/bin/env python3
""""Vanilla" Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates an autoencoder:
    input_dims is an integer containing the dimensions of the model input.
    hidden_layers is a list containing the number of nodes for each hidden
        layer in the encoder, respectively.
    the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
        representation.
    Returns: encoder, decoder, auto
    encoder is the encoder model.
    decoder is the decoder model.
    auto is the full autoencoder model.
    The autoencoder model should be compiled using adam optimization and
        binary cross-entropy loss.
    All layers should use a relu activation except for the last layer in the
        decoder, which should use sigmoid."""
    # Encoder model
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    encoded_layer = keras.layers.Dense(latent_dims, activation='relu')(x)

    encoder = keras.Model(encoder_inputs, encoded_layer, name="encoder")

    # Decoder model
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    decoder_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

    # Autoencoder model
    autoencoder_inputs = keras.Input(shape=(input_dims,))
    encoder_output = encoder(autoencoder_inputs)
    decoder_output = decoder(encoder_output)
    auto = keras.Model(autoencoder_inputs, decoder_output, name="autoencoder")

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

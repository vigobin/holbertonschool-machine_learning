#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder:
    input_dims is an integer containing the dimensions of the model input.
    hidden_layers is a list containing the number of nodes for each
        hidden layer in the encoder, respectively.
    The hidden layers should be reversed for the decoder.
    latent_dims is an integer containing the dimensions of the
        latent space representation.
    Returns: encoder, decoder, auto
    encoder is the encoder model, which should output the
        latent representation, the mean, and the log variance,
        respectively.
    decoder is the decoder model.
    auto is the full autoencoder model.
    The autoencoder model should be compiled using adam optimization
        and binary cross-entropy loss.
    All layers should use a relu activation except for the mean and log
        variance layers in the encoder, which should use None, and the
        last layer in the decoder, which should use sigmoid."""
    encoder_inputs = tf.keras.layers.Input(shape=(input_dims,))
    x = encoder_inputs
    for units in hidden_layers:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = tf.keras.layers.Dense(latent_dims, activation=None)(x)

    # Reparameterization Trick
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(
            shape=(tf.shape(z_mean)[0], latent_dims), mean=0.0, stddev=1.0)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = tf.keras.models.Model(
        encoder_inputs, [z, z_mean, z_log_var], name="encoder")

    # Decoder model
    decoder_inputs = tf.keras.layers.Input(shape=(latent_dims,))
    x = decoder_inputs
    for units in reversed(hidden_layers):
        x = tf.keras.layers.Dense(units, activation='relu')(x)
    decoded = tf.keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = tf.keras.models.Model(decoder_inputs, decoded, name="decoder")

    # Autoencoder model
    autoencoder_inputs = tf.keras.layers.Input(shape=(input_dims,))
    z_encoded, z_mean, z_log_var = encoder(autoencoder_inputs)
    decoded_output = decoder(z_encoded)
    auto = tf.keras.models.Model(
        autoencoder_inputs, decoded_output, name="vae")

    reconstruction_loss = tf.keras.losses.binary_crossentropy(
        autoencoder_inputs, decoded_output)
    reconstruction_loss = tf.keras.backend.sum(reconstruction_loss, axis=-1)
    kl_loss = -0.5 * tf.keras.backend.sum(
        1 + z_log_var - tf.keras.backend.square(
            z_mean) - tf.keras.backend.exp(z_log_var), axis=-1)
    vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)

    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto

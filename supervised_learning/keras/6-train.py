#!/usr/bin/env python3
"""6 Train"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """trains a model using mini-batch gradient descent,
      analyze validaiton data train and use early stopping"""
    callbacks = []

    if early_stopping and validation_data is not None:
        early_stop_callback = K.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience)
        callbacks.append(early_stop_callback)

    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       validation_data=validation_data, callbacks=callbacks,
                       verbose=verbose, shuffle=shuffle)

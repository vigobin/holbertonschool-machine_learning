#!/usr/bin/env python3
"""Learning Rate Decay"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True,
                shuffle=False):
    """Train the model with learning rate decay"""
    callbacks = []

    if early_stopping and validation_data is not None:
        early_stop_callback = K.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience)
        callbacks.append(early_stop_callback)

    if learning_rate_decay and validation_data is not None:
        def lr_scheduler(epoch, lr):
            """Learning rate scheduler function for inverse time decay"""
            return alpha / (1 + decay_rate * epoch)

        lr_decay_callback = K.callbacks.LearningRateScheduler(lr_scheduler,
                                                              verbose=1)
        callbacks.append(lr_decay_callback)

    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       validation_data=validation_data, callbacks=callbacks,
                       verbose=verbose, shuffle=shuffle)

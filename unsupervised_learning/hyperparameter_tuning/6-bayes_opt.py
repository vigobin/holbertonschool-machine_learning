#!/usr/bin/env python3
"""Bayesian Optimization with GPyOpt"""

import numpy as np
import GPyOpt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
Y = data.target

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2,
                                                  random_state=42)


def optimize_Gp(hyperparameters):
    """Script that optimizes a machine learning model using GPyOpt:
       Optimize at least 5 different hyperparameters.
            E.g. learning rate, number of units in a layer, dropout rate,
                L2 regularization weight, batch size.
        Your model should be optimized on a single satisficing metric.
        Your model should save a checkpoint of its best iteration during
            each training session.
        The filename of the checkpoint should specify the values of the
            hyperparameters being tuned.
        Your model should perform early stopping.
        Bayesian optimization should run for a maximum of 30 iterations.
        Once optimization has been performed, your script should plot
            the convergence.
        Your script should save a report of the optimization to the file
            'bayes_opt.txt'."""
    learning_rate, num_units, dropout_rate, l2_weight, batch = hyperparameters

    model = Sequential()
    model.add(Dense(num_units, input_dim=4, activation='relu',
                    kernel_regularizer='12', kernel_initializer='he_normal'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))

    optimizer = Adam(learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint(
        f"model_lr_{learning_rate}_units_{num_units}_dropout_{
            dropout_rate}_l2_{l2_weight}_batch_{batch}.h5",
        monitor='val_accuracy', save_best_only=True, mode='max')

    
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
        "model_lr_{}_units_{}_dropout_{}_l2_{}_batch_{}.h5".format(
            learning_rate, num_units, dropout_rate, l2_weight, batch
        ),
        monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10,
                                   mode='max')

    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                        batch_size=batch, epochs=100,
                        callbacks=[checkpoint, early_stopping], verbose=0)

    val_acc = max(history.history['val_accuracy'])
    return -val_acc  # Minimize the negative validation accuracy

# Define the hyperparameter space for optimization


space = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.1)},
    {'name': 'num_units', 'type': 'discrete', 'domain': (32, 64, 128)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'l2_weight', 'type': 'continuous', 'domain': (1e-6, 0.1)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64)}
]

# Initialize the Bayesian optimization
optimizer = GPyOpt.methods.BayesianOptimization(f=optimize_Gp, domain=space)

# Run Bayesian optimization for a maximum of 30 iterations
optimizer.run_optimization(max_iter=30)

# Save a report of the optimization
with open('bayes_opt.txt', 'w') as report_file:
    report_file.write(str(optimizer.X))
    report_file.write('\n')
    report_file.write(str(optimizer.Y))

# Plot the convergence
optimizer.plot_convergence()
plt.show()

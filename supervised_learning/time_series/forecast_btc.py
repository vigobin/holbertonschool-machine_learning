#!/usr/bin/env python3
"""When to Invest"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split


def create_dataset(data, look_back=24):
    """Create dataset"""
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back)])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

def main():
    """Calls create dataset"""
    # Load the preprocessed data
    coinbase_data = pd.read_csv('coinbase_preprocessed.csv', index_col='Timestamp', parse_dates=True)
    bitstamp_data = pd.read_csv('bitstamp_preprocessed.csv', index_col='Timestamp', parse_dates=True)

    # Create the dataset
    X, Y = create_dataset(coinbase_data.values)

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Convert the data to tf.data.Dataset
    train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    test_data = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

    # Create the model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss=MeanSquaredError())

    # Train the model
    history = model.fit(train_data.batch(10), epochs=10, verbose=2, validation_data=test_data.batch(10))

    # Plot the training loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()

    # Evaluate the model
    test_loss = model.evaluate(test_data.batch(10))
    print(f'Test Loss: {test_loss}')

if __name__ == "__main__":
    main()

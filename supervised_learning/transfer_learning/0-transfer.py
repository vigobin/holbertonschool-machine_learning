#!/usr/bin/env python3
"""Transfer Knowledge"""

import tensorflow.keras as K
import numpy as np


def preprocess_data(X, Y):
    # Scale the pixel values to [0, 1]
    X_p = X.astype('float32') / 255.0

    # One-hot encode the labels
    Y_p = K.utils.to_categorical(Y, num_classes=10)

    return X_p, Y_p


def lenet5(X):
    # Build the LeNet-5 architecture
    conv1 = K.layers.Conv2D(6, (5, 5), activation='relu', padding='same')(X)
    pool1 = K.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(16, (5, 5), activation='relu', padding='valid')(
        pool1)
    pool2 = K.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2)
    flatten = K.layers.Flatten()(pool2)
    dense1 = K.layers.Dense(120, activation='relu')(flatten)
    dense2 = K.layers.Dense(84, activation='relu')(dense1)
    output = K.layers.Dense(10, activation='softmax')(dense2)

    return output


def main():
    # Load the CIFAR 10 data
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # Preprocess the data
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Create the input layer
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    X_input = K.Input(shape=input_shape)

    # Create the LeNet-5 model
    output = lenet5(X_input)

    # Create the model
    model = K.Model(inputs=X_input, outputs=output)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # Train the model
    model.fit(
        X_train, Y_train, batch_size=32,
        epochs=10, validation_data=(X_test, Y_test))

    # Save the model
    model.save('cifar10.h5')


if __name__ == "__main__":
    main()

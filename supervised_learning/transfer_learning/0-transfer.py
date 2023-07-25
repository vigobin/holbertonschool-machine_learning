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


def densenet121():
    # Build the DenseNet-121 architecture
    input_shape = (224, 224, 3)
    base_model = K.applications.DenseNet121(
        include_top=False, weights='imagenet', input_shape=input_shape)

    # Freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    X = base_model.output
    X = K.layers.GlobalAveragePooling2D()(X)
    X = K.layers.Dense(120, activation='relu')(X)
    X = K.layers.Dense(84, activation='relu')(X)
    output = K.layers.Dense(10, activation='softmax')(X)

    return output


def main():
    # Load the CIFAR 10 data
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # Preprocess the data
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Resize the images to match DenseNet-121 input shape (224x224)
    X_train_resized = np.array([K.backend.resize_images(
        img, (224, 224)) for img in X_train])
    X_test_resized = np.array([K.backend.resize_images(
        img, (224, 224)) for img in X_test])

    # Create the input layer
    input_shape = (224, 224, 3)
    X_input = K.Input(shape=input_shape)

    # Create the DenseNet-121 model
    output = densenet121()

    # Create the model
    model = K.Model(inputs=X_input, outputs=output)

    # Compile the model
    model.compile(
        optimizer='adam', loss='categorical_crossentropy',
        metrics=['accuracy'])

    # Train the model
    model.fit(
        X_train_resized, Y_train, batch_size=32, epochs=10,
        validation_data=(X_test_resized, Y_test))

    # Save the model
    model.save('cifar10.h5')


if __name__ == "__main__":
    main()

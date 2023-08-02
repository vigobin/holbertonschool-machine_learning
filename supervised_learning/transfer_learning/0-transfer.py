#!/usr/bin/env python3
"""
Script to train a convolutional neural network to classify the CIFAR 10 dataset
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model

    parameters:
        X [numpy.ndarray of shape (m, 32, 32, 3)]:
            contains the CIFAR 10 data where m is the number of data points
        Y [numpy.ndarray of shape (m,)]:
            contains the CIFAR 10 labels for X

    returns:
        X_p: a numpy.ndarray containing the preprocessed X
        Y_p: a numpy.ndarray containing the preprocessed Y
    """
    # print(X.shape)
    # print(Y.shape)
    # scale pixels between 0 and 1
    # each channel normalized with respect to ImageNet data
    X_p = K.applications.efficientnet_v2.preprocess_input(
        X, data_format="channels_last")
    Y_p = K.utils.to_categorical(Y, 10)
    return (X_p, Y_p)


if __name__ == '__main__':
    """
    Trains a convolutional neural network to classify CIFAR 10 dataset
    Saves model to cifar10.h5
    """
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    inputs = K.Input(shape=(32, 32, 3))
    inputs_resized = K.layers.Lambda(
        lambda x: K.backend.resize_images(x,
                                          height_factor=(224 // 32),
                                          width_factor=(224 // 32),
                                          data_format="channels_last"))(inputs)

    EfficientNetV2B3 = K.applications.EfficientNetV2B3(
        include_top=False, weights='imagenet',
        input_shape=(224, 224, 3))

    X = EfficientNetV2B3(inputs_resized, training=False)
    X = K.layers.Flatten()(X)
    outputs = K.layers.Dense(10, activation='softmax')(X)

    model = K.Model(inputs=inputs, outputs=outputs)

    EfficientNetV2B3.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])

    history = model.fit(x=X_train, y=Y_train,
                        validation_data=(X_test, Y_test),
                        batch_size=600,
                        epochs=6)

    model.save('cifar10.h5')

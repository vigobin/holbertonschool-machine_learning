#!/usr/bin/env python3
"""LeNet-5 (Tensorflow)"""

import tensorflow as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using TensorFlow

    x: tf.placeholder of shape (m, 28, 28, 1) containing the input images
    for the network.
    y: tf.placeholder of shape (m, 10) containing the one-hot labels
    for the network.

    Returns:
        - tensor for the softmax activated output
        - training operation that utilizes Adam optimization
        - tensor for the loss of the network
        - tensor for the accuracy of the network
    """

    # Convolutional layer 1
    conv1 = tf.layers.Conv2D(
        filters=6, kernel_size=(
         5, 5), padding='same',
        activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.
        variance_scaling_initializer())(x)
    # Max pooling layer 1
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # Convolutional layer 2
    conv2 = tf.layers.Conv2D(
        filters=16, kernel_size=(
         5, 5), padding='valid', activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(

        ))(pool1)
    # Max pooling layer 2
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Flatten the output of the previous layer
    flatten = tf.layers.Flatten()(pool2)

    # Fully connected layer 1
    fc1 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                          kernel_initializer=tf.contrib.layers.
                          variance_scaling_initializer())(flatten)

    # Fully connected layer 2
    fc2 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                          kernel_initializer=tf.contrib.layers.
                          variance_scaling_initializer())(fc1)

    # Output layer
    output = tf.layers.Dense(units=10, activation=tf.nn.softmax,
                             kernel_initializer=tf.contrib.layers.
                             variance_scaling_initializer())(fc2)

    # Loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y, logits=output))

    # Accuracy
    correct_predictions = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # Training operation
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    return output, train_op, loss, accuracy

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

    weights_initializer = tf.contrib.layers.variance_scaling_initializer()
    C1 = tf.layers.Conv2D(filters=6,
                          kernel_size=(5, 5),
                          padding='same',
                          activation=tf.nn.relu,
                          kernel_initializer=weights_initializer)
    output_1 = C1(x)
    P2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    output_2 = P2(output_1)
    C3 = tf.layers.Conv2D(filters=16,
                          kernel_size=(5, 5),
                          padding='valid',
                          activation=tf.nn.relu,
                          kernel_initializer=weights_initializer)
    output_3 = C3(output_2)
    P4 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    output_4 = P4(output_3)
    output_42 = tf.layers.Flatten()(output_4)
    F5 = tf.layers.Dense(
        120,
        activation=tf.nn.relu,
        kernel_initializer=weights_initializer)
    output_5 = F5(output_42)
    F6 = tf.layers.Dense(
        84,
        activation=tf.nn.relu,
        kernel_initializer=weights_initializer)
    output_6 = F6(output_5)
    F7 = tf.layers.Dense(
        10,
        kernel_initializer=weights_initializer)
    output_7 = F7(output_6)
    softmax = tf.nn.softmax(output_7)
    loss = tf.losses.softmax_cross_entropy(y, logits=output_7)
    op = tf.train.AdamOptimizer().minimize(loss)
    y_pred = tf.math.argmax(output_7, axis=1)
    y_out = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(y_pred, y_out)
    accuracy = tf.reduce_mean(tf.cast(equality, 'float'))
    return softmax, op, loss, accuracy

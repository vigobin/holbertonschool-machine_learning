#!/usr/bin/env python3
"""Inception Network"""

import tensorflow.keras as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D, Dropout, Dense, Flatten
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """builds the inception network as described in
        Going Deeper with Convolutions (2014)
        You can assume the input data will have shape (224, 224, 3)
        All convolutions inside and outside the inception block
        should use a rectified linear activation (ReLU)"""
    input_shape = (224, 224, 3)
    input_layer = Input(shape=input_shape)

    conv1 = Conv2D(64, (
        7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D((
        3, 3), strides=(2, 2), padding='same')(conv1)

    conv2 = Conv2D(64, (1, 1), padding='same', activation='relu')(pool1)
    conv2 = Conv2D(192, (3, 3), padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv2)

    inception_3a = inception_block(pool2, [64, 96, 128, 16, 32])
    inception_3b = inception_block(inception_3a, [128, 128, 192, 32, 96])
    pool3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inception_3b)

    inception_4a = inception_block(pool3, [192, 96, 208, 16, 48])
    inception_4b = inception_block(inception_4a, [160, 112, 224, 24, 64])
    inception_4c = inception_block(inception_4b, [128, 128, 256, 24, 64])
    inception_4d = inception_block(inception_4c, [112, 144, 288, 32, 64])
    inception_4e = inception_block(inception_4d, [256, 160, 320, 32, 128])
    pool4 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inception_4e)

    inception_5a = inception_block(pool4, [256, 160, 320, 32, 128])
    inception_5b = inception_block(
        inception_5a, [384, 192, 384, 48, 128])

    pool5 = AveragePooling2D((
        7, 7), strides=(1, 1), padding='valid')(inception_5b)
    dropout = Dropout(0.4)(pool5)

    flatten = Flatten()(dropout)
    output_layer = Dense(units=1000, activation='softmax')(flatten)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

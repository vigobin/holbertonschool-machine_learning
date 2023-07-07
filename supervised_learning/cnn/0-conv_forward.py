#!/usr/bin/env python3
"""Convolutional Forward Prop"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a
        neural network.
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the
        output of the previous layer
    W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels
        for the convolution
    b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied
        to the convolution
    activation: activation function applied to the convolution
    padding: string indicating the type of padding used: "same" or "valid"
    stride: tuple of (sh, sw) containing the strides for the convolution

    Returns: the output of the convolutional layer
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil((h_prev - 1) * sh - h_prev + kh) / 2)
        pw = int(np.ceil((w_prev - 1) * sw - w_prev + kw) / 2)
        A_prev_pad = np.pad(
            A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    elif padding == "valid":
        A_prev_pad = A_prev

    h_out = int((h_prev - kh + 2 * ph) / sh) + 1
    w_out = int((w_prev - kw + 2 * pw) / sw) + 1

    A = np.zeros((m, h_out, w_out, c_new))

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            A[:, i, j, :] = np.sum(A_prev_pad[
                :, h_start:h_end, w_start:w_end,
                :, np.newaxis] * W, axis=(1, 2, 3))

    Z = A + b
    A_out = activation(Z)

    return A_out

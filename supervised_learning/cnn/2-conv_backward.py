#!/usr/bin/env python3
"""Convolutional Back Prop"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs backpropagation over a convolutional layer of a neural network

    dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the partial
    derivatives with respect to the unactivated output of the
    convolutional layer.
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the
    output of the previous layer.
    W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels
    for the convolution.
    b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied
    to the convolution.
    padding: string indicating the type of padding used: "same" or "valid"
    stride: tuple of (sh, sw) containing the strides for the convolution.

    Returns: the partial derivatives with respect to the previous layer
    (dA_prev), the kernels (dW), and the biases (db), respectively.
    """

    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, _ = W.shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == "same":
        ph = max((A_prev.shape[1] - 1) * sh - A_prev.shape[1] + kh, 0)
        pw = max((A_prev.shape[2] - 1) * sw - A_prev.shape[2] + kw, 0)
        pad_top = ph // 2
        pad_bottom = ph - pad_top
        pad_left = pw // 2
        pad_right = pw - pad_left
        A_prev_pad = np.pad(A_prev, ((0, 0), (pad_top, pad_bottom), (
            pad_left, pad_right), (0, 0)), mode='constant')
        dA_prev_pad = np.pad(dA_prev, ((0, 0), (pad_top, pad_bottom), (
            pad_left, pad_right), (0, 0)), mode='constant')
    elif padding == "valid":
        A_prev_pad = A_prev
        dA_prev_pad = dA_prev

    for i in range(h_new):
        for j in range(w_new):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            dA_prev_pad[:, h_start:h_end, w_start:w_end, :] += np.sum(
                W * dZ[:, i:i+1, j:j+1, :], axis=-1)
            dW += np.sum(A_prev_pad[
                :, h_start:h_end, w_start:w_end, :, np.newaxis] * dZ[
                    :, i:i+1, j:j+1, np.newaxis, :], axis=0)

    if padding == "same":
        dA_prev = dA_prev_pad[:, pad_top:-pad_bottom, pad_left:-pad_right, :]
    elif padding == "valid":
        dA_prev = dA_prev_pad

    return dA_prev, dW, db

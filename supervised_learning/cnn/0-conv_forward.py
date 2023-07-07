#!/usr/bin/env python3
"""Pooling Forward Prop"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network

    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
    kernel_shape: tuple of (kh, kw) containing the size of the kernel for
        pooling
    stride: tuple of (sh, sw) containing the strides for pooling
    mode: string containing either 'max' or 'avg', indicating whether
        to perform maximum or average pooling

    Returns: the output of the pooling layer
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_out = int((h_prev - kh) / sh) + 1
    w_out = int((w_prev - kw) / sw) + 1

    A = np.zeros((m, h_out, w_out, c_prev))

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            if mode == 'max':
                A[:, i, j, :] = np.max(
                    A_prev[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(
                    A_prev[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))

    return A

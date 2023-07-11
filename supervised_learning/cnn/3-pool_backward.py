#!/usr/bin/env python3
"""Pooling Back Prop"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backpropagation over a pooling layer of a neural network

    dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing
    the partial derivatives with respect to the output of the pooling layer.
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
    output of the previous layer.
    kernel_shape: tuple of (kh, kw) containing the size of the kernel
    for the pooling.
    stride: tuple of (sh, sw) containing the strides for the pooling
    mode: string indicating the pooling mode: "max" or "avg"

    Returns: the partial derivatives with respect to the
    previous layer (dA_prev).
    """

    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    if mode == "max":
                        mask = (A_prev[
                            i, h_start:h_end, w_start:w_end, c] == np.max(
                            A_prev[i, h_start:h_end, w_start:w_end, c]))
                        dA_prev[
                            i, h_start:h_end, w_start:w_end, c] += mask * dA[
                                i, h, w, c]
                    elif mode == "avg":
                        average = dA[i, h, w, c] / (kh * kw)
                        dA_prev[i, h_start:h_end, w_start:w_end, c] += np.ones(
                            (kh, kw)) * average

    return dA_prev

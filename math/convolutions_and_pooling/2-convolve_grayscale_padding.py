#!/usr/bin/env python3
"""Convolution with Padding """

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """performs a convolution on grayscale images with custom padding"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh, kw = kernel.shape
    ph, pw = padding

    # Calculate the output shape
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    # Initialize the output array
    output = np.zeros((m, output_h, output_w))

    # Pad the images
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')

    # Perform the convolution
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(padded_images[:, i:i+kh, j:j+kw] * kernel,
                                     axis=(1, 2))

    return output

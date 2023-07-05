#!/usr/bin/env python3
"""Multiple Kernels"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """performs a convolution on images using multiple kernels"""
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    # Calculate the output shape based on the padding and stride
    if padding == 'same':
        pad_h = max((h - 1) * sh + kh - h, 0)
        pad_w = max((w - 1) * sw + kw - w, 0)
    elif padding == 'valid':
        pad_h, pad_w = 0, 0
    else:
        pad_h, pad_w = padding

    # Pad the images
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w),
                                    (0, 0)), mode='constant')

    # Calculate the output shape
    output_h = (h + 2 * pad_h - kh) // sh + 1
    output_w = (w + 2 * pad_w - kw) // sw + 1

    # Initialize the output array
    output = np.zeros((m, output_h, output_w, nc))

    # Perform the convolution
    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):
                output[:, i, j, k] = np.sum(
                    padded_images[
                        :, i*sh:i*sh+kh, j*sw:j*sw+kw, :] * kernels[
                            :, :, :, k], axis=(1, 2, 3))

    return output

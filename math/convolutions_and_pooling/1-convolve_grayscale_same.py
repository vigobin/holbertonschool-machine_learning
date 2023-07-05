#!/usr/bin/env python3
"""Same Convolution"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """that performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate the padding size
    pad_h = int((kh - 1) // 2)
    pad_w = int((kw - 1) // 2)

    # Pad the images
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant')

    # Initialize the output array
    output = np.zeros_like(images)

    # Perform the convolution
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(padded_images[:, i:i+kh, j:j+kw] * kernel,
                                     axis=(1, 2))

    return output

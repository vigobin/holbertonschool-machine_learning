#!/usr/bin/env python3
"""Same Convolution"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """that performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding sizes
    ph = int(np.ceil(((h - 1) * 1 + kh - h) / 2))
    pw = int(np.ceil(((w - 1) * 1 + kw - w) / 2))

    # Create padded images
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')

    # Initialize the output array
    output = np.zeros((m, h, w))

    # Perform convolution
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(padded_images[
                :, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return output

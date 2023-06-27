#!/usr/bin/env python3
"""Create Confusion"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix"""
    return np.matmul(labels.transpose(), logits)

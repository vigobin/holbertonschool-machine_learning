#!/usr/bin/env python3
"""Sensitivity function"""

import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix"""
    # Calculate the number of classes
    classes = confusion.shape[0]
    # Initialize an array to store the sensitivity values
    sensitivity_values = np.zeros(classes)
    for i in range(classes):
        # Calculate the true positives for class i
        true_positives = confusion[i, i]
        # Calculate the sum of true positives and false negatives for class i
        positives = np.sum(confusion[i, :])
        # Calculate the sensitivity for class i
        sensitivity_values[i] = true_positives / positives
    return sensitivity_values

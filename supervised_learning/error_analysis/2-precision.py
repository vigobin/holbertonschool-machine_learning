#!/usr/bin/env python3
"""Precision function"""

import numpy as np


def precision(confusion):
    """Calculates the precision for each class in a confusion matrix"""
    # Calculate the number of classes
    classes = confusion.shape[0]
    # Initialize an array to store the precision values
    precision_values = np.zeros(classes)
    for i in range(classes):
        # Calculate the true positives for class i
        true_positives = confusion[i, i]
        # Calculate the sum of true positives and false positives for class i
        predicted_positives = np.sum(confusion[:, i])
        # Calculate the precision for class i
        precision_values[i] = true_positives / predicted_positives
    return precision_values

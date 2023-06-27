#!/usr/bin/env python3
"""Specificity function"""

import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class in a confusion matrix"""
    # Calculate the number of classes
    classes = confusion.shape[0]
    # Initialize an array to store the specificity values
    specificity_values = np.zeros(classes)
    for i in range(classes):
        # Calculate the true negatives for class i
        true_negatives = np.sum(np.delete(np.delete(
            confusion, i, axis=0), i, axis=1))
        # Calculate the sum of true negatives and false positives for class i
        negatives = np.sum(np.delete(confusion[i, :], i))
        # Calculate the specificity for class i
        specificity_values[i] = true_negatives / negatives

    return specificity_values

#!/usr/bin/env python3
"""F1 Score"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the F1 score of a confusion matrix"""
    # Calculate the number of classes
    classes = confusion.shape[0]
    # Calculate the sensitivity and precision for each class
    sensitivities = sensitivity(confusion)
    precisions = precision(confusion)
    # Initialize an array to store the F1 scores
    f1_scores = np.zeros(classes)
    for i in range(classes):
        # Calculate the F1 score for class i
        f1_scores[i] = 2 * (precisions[i] * sensitivities[i]) / (
            precisions[i] + sensitivities[i])

    return f1_scores

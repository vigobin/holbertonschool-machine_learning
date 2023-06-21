#!/usr/bin/env python3
"""Moving Average"""


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set"""

    v = 0
    EMA = []
    for i in range(len(data)):
        v = ((v * beta) + ((1 - beta) * data[i]))
        EMA.append(v / (1 - (beta ** (i + 1))))

    return EMA

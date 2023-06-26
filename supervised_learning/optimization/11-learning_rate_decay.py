#!/usr/bin/env python3
"""Learning Rate Decay"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates the learning rate using inverse time decay in numpy"""
    lr = alpha/(1 + decay_rate * (global_step//decay_step))
    return lr

#!/usr/bin/env python3
"""Bayesian Optimization with GPyOpt"""

import numpy as np
import GPyOpt


def optimize_Gp():
    """ Sript that optimizes a machine learning model of your choice using GPyOpt:
        Your script should optimize at least 5 different hyperparameters. E.g. learning rate, number of units in a layer, dropout rate, L2 regularization weight, batch size
        Your model should be optimized on a single satisficing metric
        Your model should save a checkpoint of its best iteration during each training session
        The filename of the checkpoint should specify the values of the hyperparameters being tuned
        Your model should perform early stopping
        Bayesian optimization should run for a maximum of 30 iterations
        Once optimization has been performed, your script should plot the convergence
        Your script should save a report of the optimization to the file 'bayes_opt.txt'."""

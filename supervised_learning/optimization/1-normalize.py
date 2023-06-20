#!/usr/bin/env python3
"""Normalize matrix"""


def normalize(X, m, s):
    """Normalizes (standardizes) a matrix"""
    return ((X - m) / s)

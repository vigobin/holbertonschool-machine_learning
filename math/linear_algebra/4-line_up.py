#!/usr/bin/env python3
"""Function that adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """Returns a new list with the result"""
    if len(arr1) != len(arr2):
        return None
    sum_list = []
    for i in range(len(arr1)):
        sum_list.append(arr1[i] + arr2[i])
    return sum_list

#!/usr/bin/env python3
"""Function that concatenates two arrays"""


def cat_arrays(arr1, arr2):
    """Returns a new list with concatenated arrays"""
    new_list = []
    for i in arr1:
        new_list.append(i)
    for i in arr2:
        new_list.append(i)
    return new_list

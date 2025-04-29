"""
Utils.py
"""

import numpy as np
import math

def shift(arr, len, fill_value=0.0):
    return np.pad(arr, (len,), mode='constant', constant_values=(fill_value,))[:arr.size]


def barssince(s: np.array):
    val = np.array([0.0]*s.size)
    c = math.nan
    for i in range(s.size):
        if s[i]: c = 0; continue
        if c >= 0: c += 1
        val[i] = c
    return val

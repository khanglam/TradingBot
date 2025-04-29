import math
import pandas as pd


def rationalQuadratic(src: pd.Series, lookback: int, relativeWeight: float, startAtBar: int):
    """
    vectorized calculate for rational quadratic curve
    :param src:
    :param lookback:
    :param relativeWeight:
    :param startAtBar:
    :return:
    """
    currentWeight = [0.0]*len(src)
    cumulativeWeight = 0.0
    for i in range(startAtBar + 2):
        y = src.shift(i, fill_value=0.0)
        w = (1 + (i ** 2 / (lookback ** 2 * 2 * relativeWeight))) ** -relativeWeight
        currentWeight += y.values * w
        cumulativeWeight += w
    val = currentWeight / cumulativeWeight
    val[:startAtBar + 1] = 0.0

    return val


def gaussian(src: pd.Series, lookback: int, startAtBar: int):
    """
    vectorized calculate for gaussian curve
    :param src:
    :param lookback:
    :param startAtBar:
    :return:
    """
    currentWeight = [0.0]*len(src)
    cumulativeWeight = 0.0
    for i in range(startAtBar + 2):
        y = src.shift(i, fill_value=0.0)
        w = math.exp(-(i ** 2) / (2 * lookback ** 2))
        currentWeight += y.values * w
        cumulativeWeight += w
    val = currentWeight / cumulativeWeight
    val[:startAtBar + 1] = 0.0

    return val

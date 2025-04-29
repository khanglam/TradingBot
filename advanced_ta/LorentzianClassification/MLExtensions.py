import math
import numpy as np
import pandas as pd
from ta.momentum import rsi as RSI
from ta.volatility import average_true_range as ATR
from ta.trend import cci as CCI, adx as ADX, ema_indicator as EMA, sma_indicator as SMA
from sklearn.preprocessing import MinMaxScaler


# ==========================
# ==== Helper Functions ====
# ==========================


def normalize(src: np.array, range_min=0, range_max=1) -> np.array:
    """
    function Rescales a source value with an unbounded range to a bounded range
    param src: <np.array> The input series
    param range_min: <float> The minimum value of the unbounded range
    param range_max: <float> The maximum value of the unbounded range
    returns <np.array> The normalized series
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return range_min + (range_max - range_min) * scaler.fit_transform(src.reshape(-1,1))[:,0]


def rescale(src: np.array, old_min, old_max, new_min=0, new_max=1) -> np.array:
    """
    function Rescales a source value with a bounded range to anther bounded range
    param src: <np.array> The input series
    param old_min: <float> The minimum value of the range to rescale from
    param old_max: <float> The maximum value of the range to rescale from
    param new_min: <float> The minimum value of the range to rescale to
    param new_max: <float> The maximum value of the range to rescale to 
    returns <np.array> The rescaled series
    """
    rescaled_value = new_min + (new_max - new_min) * (src - old_min) / max(old_max - old_min, 10e-10)
    return rescaled_value


def n_rsi(src: pd.Series, n1, n2) -> np.array:
    """
    function Returns the normalized RSI ideal for use in ML algorithms
    param src: <np.array> The input series
    param n1: <int> The length of the RSI
    param n2: <int> The smoothing length of the RSI
    returns <np.array> The normalized RSI
    """
    return rescale(EMA(RSI(src, n1), n2).values, 0, 100)


def n_cci(highSrc: pd.Series, lowSrc: pd.Series, closeSrc: pd.Series, n1, n2) -> np.array:
    """
    function Returns the normalized CCI ideal for use in ML algorithms
    param highSrc: <np.array> The input series for the high price
    param lowSrc: <np.array> The input series for the low price
    param closeSrc: <np.array> The input series for the close price
    param n1: <int> The length of the CCI
    param n2: <int> The smoothing length of the CCI
    returns <np.array> The normalized CCI
    """
    return normalize(EMA(CCI(highSrc, lowSrc, closeSrc, n1), n2).values)

def n_wt(src: pd.Series, n1=10, n2=11) -> np.array:
    """
    function Returns the normalized WaveTrend Classic series ideal for use in ML algorithms
    param src: <np.array> The input series
    param n1: <int> The first smoothing length for WaveTrend Classic
    param n2: <int> The second smoothing length for the WaveTrend Classic
    returns <np.array> The normalized WaveTrend Classic series
    """
    ema1 = EMA(src, n1)
    ema2 = EMA(abs(src - ema1), n1)
    ci = (src - ema1) / (0.015 * ema2)
    wt1 = EMA(ci, n2)  # tci
    wt2 = SMA(wt1, 4)
    return normalize((wt1 - wt2).values)

def n_adx(highSrc: pd.Series, lowSrc: pd.Series, closeSrc: pd.Series, n1) -> np.array:
    """
    function Returns the normalized ADX ideal for use in ML algorithms
    param highSrc: <np.array> The input series for the high price
    param lowSrc: <np.array> The input series for the low price
    param closeSrc: <np.array> The input series for the close price
    param n1: <int> The length of the ADX
    """
    return rescale(ADX(highSrc, lowSrc, closeSrc, n1).values, 0, 100)
    # TODO: Replicate ADX logic from jdehorty


# =================
# ==== Filters ====
# =================
def regime_filter(src: pd.Series, high: pd.Series, low: pd.Series, useRegimeFilter, threshold) -> np.array:
    """
    regime_filter
    param src: <np.array> The source series
    param high: <np.array> The input series for the high price
    param low: <np.array> The input series for the low price
    param useRegimeFilter: <bool> Whether to use the regime filter
    param threshold: <float> The threshold
    returns <np.array> Boolean indicating whether or not to let the signal pass through the filter
    """
    if not useRegimeFilter: return np.array([True]*len(src))

    # @njit(parallel=True, cache=True)
    def klmf(src: np.array, high: np.array, low: np.array):
        value1 = np.array([0.0]*len(src))
        value2 = np.array([0.0]*len(src))
        klmf = np.array([0.0]*len(src))

        for i in range(len(src)):
            if (high[i] - low[i]) == 0: continue
            value1[i] = 0.2 * (src[i] - src[i - 1 if i >= 1 else 0]) + 0.8 * value1[i - 1 if i >= 1 else 0]
            value2[i] = 0.1 * (high[i] - low[i]) + 0.8 * value2[i - 1 if i >= 1 else 0]

        with np.errstate(divide='ignore',invalid='ignore'):
            omega = np.nan_to_num(np.abs(np.divide(value1, value2)))
        alpha = (-(omega ** 2) + np.sqrt((omega ** 4) + 16 * (omega ** 2))) / 8

        for i in range(len(src)):
            klmf[i] = alpha[i] * src[i] + (1 - alpha[i]) * klmf[i - 1 if i >= 1 else 0]

        return klmf

    filter = np.array([False]*len(src))
    absCurveSlope = np.abs(np.diff(klmf(src.values, high.values, low.values), prepend=0.0))
    exponentialAverageAbsCurveSlope = EMA(pd.Series(absCurveSlope), 200).values
    with np.errstate(divide='ignore',invalid='ignore'):
        normalized_slope_decline = (absCurveSlope - exponentialAverageAbsCurveSlope) / exponentialAverageAbsCurveSlope
    flags = (normalized_slope_decline >= threshold)
    filter[(len(filter) - len(flags)):] = flags
    return filter

def filter_adx(src: pd.Series, high: pd.Series, low: pd.Series, adxThreshold, useAdxFilter, length=14) -> np.array:
    """
    function filter_adx
    param src: <np.array> The source series
    param high: <np.array> The input series for the high price
    param low: <np.array> The input series for the low price
    param adxThreshold: <int> The ADX threshold
    param useAdxFilter: <bool> Whether to use the ADX filter
    param length: <int> The length of the ADX
    returns <np.array> Boolean indicating whether or not to let the signal pass through the filter
    """
    if not useAdxFilter: return np.array([True]*len(src))
    adx = ADX(high, low, src, length).values
    return (adx > adxThreshold)

def filter_volatility(high, low, close, useVolatilityFilter, minLength=1, maxLength=10) -> np.array:
    """
    function filter_volatility
    param high: <np.array> The input series for the high price
    param low: <np.array> The input series for the low price
    param close: <np.array> The input series for the close price
    param useVolatilityFilter: <bool> Whether to use the volatility filter
    param minLength: <int> The minimum length of the ATR
    param maxLength: <int> The maximum length of the ATR
    returns <np.array> Boolean indicating whether or not to let the signal pass through the filter
    """
    if not useVolatilityFilter: return np.array([True]*len(close))
    recentAtr = ATR(high, low, close, minLength).values
    historicalAtr = ATR(high, low, close, maxLength).values
    return (recentAtr > historicalAtr)

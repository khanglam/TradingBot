import logging
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from enum import IntEnum
import mplfinance as mpf
import yfinance as yf
import traceback
import matplotlib.pyplot as plt


# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    filename='application.log',  # Name of the log file
    filemode='w',  # 'w' to overwrite the log file on each run, 'a' to append
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# The following functions are custom implementations of popular technical indicators used in trading.
# These indicators are typically used to analyze financial markets and make trading decisions.

def custom_rsi(data: pd.Series, period: int) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).
    RSI is a momentum oscillator that measures the speed and change of price movements.
    
    Args:
    data (pd.Series): A pandas Series of prices (usually closing prices).
    period (int): The number of periods to use for calculation (typically 14).
    
    Returns:
    pd.Series: A pandas Series containing the RSI values.
    """
    logging.info("Calculating RSI for period: %d", period)
    # Calculate the difference between consecutive prices
    delta = data.diff()
    
    # Isolate the gains and losses from the delta
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Calculate the Relative Strength (RS)
    rs = gain / loss
    
    # Calculate the RSI using the formula
    rsi = 100 - (100 / (1 + rs))
    logging.debug("RSI values: %s", rsi.tail())
    return rsi

def custom_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Calculates the Commodity Channel Index (CCI).
    CCI identifies cyclical trends and is often used to detect overbought and oversold levels.
    
    Args:
    high (pd.Series): High prices.
    low (pd.Series): Low prices.
    close (pd.Series): Closing prices.
    period (int): The number of periods to use for calculation.
    
    Returns:
    pd.Series: A pandas Series containing the CCI values.
    """
    logging.info("Calculating CCI for period: %d", period)
    # Calculate the typical price for each period
    tp = (high + low + close) / 3
    
    # Calculate CCI using its formula
    cci = (tp - tp.rolling(window=period).mean()) / (0.015 * tp.rolling(window=period).std())
    logging.debug("CCI values: %s", cci.tail())
    return cci

def custom_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Calculates the Average Directional Index (ADX).
    ADX is used to quantify trend strength and is non-directional; it does not indicate trend direction.
    
    Args:
    high (pd.Series): High prices.
    low (pd.Series): Low prices.
    close (pd.Series): Closing prices.
    period (int): The number of periods to use for calculation.
    
    Returns:
    pd.Series: A pandas Series containing the ADX values.
    """
    logging.info("Calculating ADX for period: %d", period)
    # Calculate the differences in highs and lows
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    # Calculate the True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth the True Range using a rolling sum
    tr_s = tr.rolling(window=period).sum()
    
    # Clean up and prepare directional movements for calculation
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = minus_dm.abs()
    plus_dm_s = plus_dm.rolling(window=period).sum()
    minus_dm_s = minus_dm.rolling(window=period).sum()
    
    # Calculate the Positive and Negative Directional Indicators
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # Calculate the ADX
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()
    logging.debug("ADX values: %s", adx.tail())
    return adx

# Helper Functions from MLExtensions.py
def normalize(src: pd.Series, min, max) -> pd.Series:
    logging.info("Normalizing data series with min: %f, max: %f", min, max)
    scaler = MinMaxScaler(feature_range=(min, max))
    
    # Reset the index to default integer index before scaling
    src_reset = src.reset_index(drop=True)
    normalized_data = pd.Series(
        data=min + (max - min) * scaler.fit_transform(pd.DataFrame({'data': src_reset})).squeeze(),  # Squeeze to get a Series
        index=src.index  # Use the original index
    )

    logging.debug("Normalized values: %s", normalized_data.tail())
    return normalized_data
    
def rescale(src: pd.Series, old_min, old_max, new_min, new_max) -> pd.Series:
    """
    Rescales a pandas Series from one range to another.
    
    Args:
    src (pd.Series): The input data series to rescale.
    old_min, old_max (float): The original range of the data.
    new_min, new_max (float): The target range for rescaling.
    
    Returns:
    pd.Series: Rescaled data series.
    """
    # Perform the rescaling calculation
    logging.info("Rescaling data series from range [%f, %f] to [%f, %f]", old_min, old_max, new_min, new_max)
    rescaled_value = new_min + (new_max - new_min) * (src - old_min) / max(old_max - old_min, 10e-10)
    logging.debug("Rescaled values: %s", rescaled_value.tail())
    return rescaled_value
    
def RMA(df: pd.Series, length: int) -> pd.Series:
    """
    Calculates the Running Moving Average (RMA).
    
    Args:
    df (pd.Series): The input data series.
    len (int): The number of periods to consider for the average.
    
    Returns:
    pd.Series: The running moving average of the input data.
    """
    # Copy the input series and apply the RMA calculation
    logging.info("Calculating Running Moving Average (RMA) with length: %d", length)
    rma = df.copy()
    rma.iloc[:length] = rma.rolling(length).mean().iloc[:length]
    rma = rma.ewm(alpha=(1.0/length), adjust=False).mean()
    logging.debug("RMA values: %s", rma.tail())
    return rma

def n_rsi(src: pd.Series, n1, n2) -> pd.Series:
    """
    Normalized Relative Strength Index (RSI).
    
    Args:
    src (pd.Series): The input data series (typically closing prices).
    n1, n2 (int): Parameters for the RSI calculation.
    
    Returns:
    pd.Series: Normalized RSI values.
    """
    logging.info("Calculating Normalized RSI with parameters: n1=%d, n2=%d", n1, n2)
    normalized_rsi = rescale(custom_rsi(src, n1), 0, 100, 0, 1)
    logging.debug("Normalized RSI values: %s", normalized_rsi.tail())
    return normalized_rsi

def n_cci(highSrc: pd.Series, lowSrc: pd.Series, closeSrc: pd.Series, n1, n2) -> pd.Series:
    logging.info("Calculating Normalized CCI with parameters: n1=%d, n2=%d", n1, n2)
    normalized_cci = normalize(custom_cci(highSrc, lowSrc, closeSrc, n1), 0, 1)
    logging.debug("Normalized CCI values: %s", normalized_cci.tail())
    return normalized_cci

def custom_ema(series: pd.Series, period: int) -> pd.Series:
    logging.info("Calculating Exponential Moving Average (EMA) with period: %d", period)
    ema = series.ewm(span=period, adjust=False).mean()
    logging.debug("EMA values: %s", ema.tail())
    return ema

def custom_sma(series: pd.Series, period: int) -> pd.Series:
    logging.info("Calculating Simple Moving Average (SMA) with period: %d", period)
    sma = series.rolling(window=period).mean()
    logging.debug("SMA values: %s", sma.tail())
    return sma

def n_wt(src: pd.Series, n1=10, n2=11) -> pd.Series:
    logging.info("Calculating Normalized WT with parameters: n1=%d, n2=%d", n1, n2)
    ema1 = custom_ema(src, n1)
    ema2 = custom_ema(abs(src - ema1), n1)
    ci = (src - ema1) / (0.015 * ema2)
    wt1 = custom_ema(ci, n2)
    wt2 = custom_sma(wt1, 4)
    wt_difference = wt1 - wt2
    logging.debug("WT difference values before normalization: %s", wt_difference.tail())
    normalized_wt = normalize(wt1 - wt2, 0, 1)
    logging.debug("Normalized WT values: %s", normalized_wt.tail())
    return normalized_wt

def n_adx(highSrc: pd.Series, lowSrc: pd.Series, closeSrc: pd.Series, n1) -> pd.Series:
    logging.info("Calculating Normalized ADX with parameter: n1=%d", n1)
    normalized_adx = rescale(custom_adx(highSrc, lowSrc, closeSrc, n1), 0, 100, 0, 1)
    logging.debug("Normalized ADX values: %s", normalized_adx.tail())
    return normalized_adx

def regime_filter(src: pd.Series, high: pd.Series, low: pd.Series, useRegimeFilter, threshold):
    """
    Applies a regime filter to identify specific market conditions.
    
    Args:
    src (pd.Series): The source data series.
    high, low (pd.Series): High and low price series.
    useRegimeFilter (bool): Flag to activate the regime filter.
    threshold (float): Threshold value for the filter.
    
    Returns:
    pd.Series: A series of boolean values indicating the regime.
    """
    logging.info("Applying regime filter with threshold: %f", threshold)
    
    if not useRegimeFilter: 
        return pd.Series(True, index=src.index)

    value1 = pd.Series([0.0] * len(src), index=src.index)
    value2 = pd.Series([0.0] * len(src), index=src.index)
    klmf = pd.Series([0.0] * len(src), index=src.index)
    absCurveSlope = pd.Series([0.0] * len(src), index=src.index)
    filter = pd.Series(False, index=src.index)

    for i in range(len(src)):
        if pd.isna(src.iloc[i]) or pd.isna(high.iloc[i]) or pd.isna(low.iloc[i]):
            # Skip calculation if NaN is encountered
            continue

        if (high.iloc[i] - low.iloc[i]) == 0:
            filter.iloc[i] = False
            continue

        # Rest of the calculations...
        value1.iloc[i] = 0.2 * (src.iloc[i] - src.iloc[i - 1 if i >= 1 else 0]) + 0.8 * value1.iloc[i - 1 if i >= 1 else 0]
        value2.iloc[i] = 0.1 * (high.iloc[i] - low.iloc[i]) + 0.9 * value2.iloc[i - 1 if i >= 1 else 0]
        omega = abs(value1.iloc[i] / value2.iloc[i])
        alpha = (-(omega ** 2) + math.sqrt((omega ** 4) + 16 * (omega ** 2))) / 8 
        klmf.iloc[i] = alpha * src.iloc[i] + (1 - alpha) * klmf.iloc[i - 1 if i >= 1 else 0]
        absCurveSlope.iloc[i] = abs(klmf.iloc[i] - klmf.iloc[i - 1 if i >= 1 else 0])
        
        # Calculate EMA of absCurveSlope
        exponentialAverageAbsCurveSlope = custom_ema(absCurveSlope, 200).iloc[i]

        # Initialize normalized_slope_decline
        normalized_slope_decline = np.nan

        # Check for valid exponentialAverageAbsCurveSlope before computing normalized_slope_decline
        if not np.isnan(exponentialAverageAbsCurveSlope) and exponentialAverageAbsCurveSlope != 0:
            normalized_slope_decline = (absCurveSlope.iloc[i] - exponentialAverageAbsCurveSlope) / exponentialAverageAbsCurveSlope
            filter.iloc[i] = normalized_slope_decline >= threshold
        else:
            filter.iloc[i] = False  # Handle NaN or zero cases by setting the filter to False

        logging.debug("At index %d, value1: %f, value2: %f, omega: %f, alpha: %f, klmf: %f, absCurveSlope: %f, exponentialAverageAbsCurveSlope: %f, normalized_slope_decline: %f, filter: %s", 
                      i, value1.iloc[i], value2.iloc[i], omega, alpha, klmf.iloc[i], absCurveSlope.iloc[i], exponentialAverageAbsCurveSlope, normalized_slope_decline, filter.iloc[i])

    logging.info("Regime filter applied successfully.")
    return filter

def filter_adx(src: pd.Series, high: pd.Series, low: pd.Series, adxThreshold, useAdxFilter, length=14):
    """
    Filters data based on the Average Directional Index (ADX) to determine the strength of a trend.
    
    Args:
    src (pd.Series): The source data series (typically closing prices).
    high, low (pd.Series): High and low price series.
    adxThreshold (float): Threshold value for ADX to determine trend strength.
    useAdxFilter (bool): Flag to activate the ADX filter.
    length (int): The number of periods to use for ADX calculation.
    
    Returns:
    pd.Series: A series of boolean values indicating whether each data point passes the ADX filter.
    """
    # Return a series of True if ADX filter is not used
    logging.info("Applying ADX filter with threshold: %f, and length: %d", adxThreshold, length)
    if not useAdxFilter: return pd.Series(True, index=src.index)

    # Calculate components for ADX calculation
    tr = np.max([high - low, np.abs(high - src.shift(1)), np.abs(low - src.shift(1))], axis=0)
    directionalMovementPlus = np.where((high - high.shift(1)) > (low.shift(1) - low), high - high.shift(1), 0)
    negMovement = np.where(low > (high - high.shift(1)), low.shift(1) - low, 0)

    # Smooth the true range and directional movements
    trSmooth = tr.rolling(window=length).mean()
    smoothDirectionalMovementPlus = directionalMovementPlus.rolling(window=length).mean()
    smoothnegMovement = negMovement.rolling(window=length).mean()

    # Calculate the Positive and Negative Directional Indicators
    diPositive = 100 * smoothDirectionalMovementPlus / trSmooth
    diNegative = 100 * smoothnegMovement / trSmooth
    logging.debug("Calculated components: TR Smooth, DI Positive, DI Negative")
    
    # Calculate the ADX and apply the threshold filter
    dx = np.abs(diPositive - diNegative) / (diPositive + diNegative) * 100
    adx = RMA(dx, length)
    logging.debug("Calculated ADX values")
    adx_filter = (adx > adxThreshold)
    # Log the decision for each data point
    for i in range(len(adx_filter)):
        logging.debug("At index %d, ADX value: %f, Filter decision: %s", i, adx.iloc[i], adx_filter.iloc[i])

        logging.info("ADX filter applied successfully.")
    return adx_filter

def custom_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Custom implementation of the Average True Range (ATR).
    ATR is a technical indicator that measures market volatility.
    
    Args:
    high, low, close (pd.Series): High, low, and closing price series.
    period (int): The number of periods to use for ATR calculation.
    
    Returns:
    pd.Series: The ATR values.
    """
    # Calculate the true range
    logging.debug("Calculating True Range...")
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate the ATR
    logging.debug("Calculating Average True Range (ATR) with period %d...", period)
    atr = tr.rolling(window=period).mean()
    logging.info("ATR calculated successfully.")
    return atr

def filter_volatility(high, low, close, useVolatilityFilter, minLength=1, maxLength=10):
    """
    Filters data based on volatility, measured using the Average True Range (ATR).
    
    Args:
    high, low, close (pd.Series): High, low, and closing price series.
    useVolatilityFilter (bool): Flag to activate the volatility filter.
    minLength, maxLength (int): Parameters to define the ATR period for volatility comparison.
    
    Returns:
    pd.Series: A series of boolean values indicating whether each data point passes the volatility filter.
    """
    # Return a series of True if volatility filter is not used
    logging.debug("Applying volatility filter...")
    if not useVolatilityFilter: return pd.Series(True, index=close.index)

    # Calculate ATR for recent and historical periods
    recentAtr = custom_atr(high, low, close, minLength)
    historicalAtr = custom_atr(high, low, close, maxLength)

    logging.debug("Comparing recent ATR with historical ATR for volatility filter.")
    volatility_filter = (recentAtr > historicalAtr)
    logging.info("Volatility filter applied successfully.")
    return volatility_filter

# Kernel Functions from KernelFunctions.py
def rationalQuadratic(src: pd.Series, lookback: int, relativeWeight: float, startAtBar: int):
    """
    Computes the rational quadratic kernel estimation for a given time series.
    
    Args:
    src (pd.Series): The input data series (usually a financial time series).
    lookback (int): The lookback period to consider for the kernel calculation.
    relativeWeight (float): The relative weighting factor for the kernel calculation.
    startAtBar (int): The starting index from which to compute the kernel.
    
    Returns:
    pd.Series: A pandas Series containing the values of the rational quadratic kernel.
    """
    logging.debug("Computing rational quadratic kernel estimation...")
    # Calculate the size of the window for kernel calculation
    size = startAtBar + 2

    # Create windows of data based on the lookback period and start index
    windows = [src[i:i + size].values for i in range(0, len(src) - size + 1)]

    # Calculate the weight for each window
    weight = [math.pow(1 + (math.pow(i, 2) / (math.pow(lookback, 2) * 2 * relativeWeight)), -relativeWeight) for i in range(size)]

    # Compute the current weight for each window
    current_weight = [np.sum(windows[i][::-1] * weight) for i in range(0, len(src) - size + 1)]

    # Calculate the cumulative weight
    cumulative_weight = [np.sum(weight) for _ in range(0, len(src) - size + 1)]

    # Compute the kernel line by normalizing the current weight with the cumulative weight
    kernel_line = np.array(current_weight) / np.array(cumulative_weight)

    # Append zeroes to match the size of the input series and create the final kernel line
    kernel_line = np.concatenate((np.array([0.0] * (size - 1)), kernel_line))
    kernel_line = pd.Series(kernel_line.flatten())
    logging.info("Rational quadratic kernel estimation completed.")
    return kernel_line
    

def gaussian(src, lookback, startAtBar):
    """
    Computes the Gaussian kernel estimation for a given time series.
    
    Args:
    src (pd.Series): The input data series (usually a financial time series).
    lookback (int): The lookback period to consider for the kernel calculation.
    startAtBar (int): The starting index from which to compute the kernel.
    
    Returns:
    pd.Series: A pandas Series containing the values of the Gaussian kernel.
    """
    logging.debug("Computing Gaussian kernel estimation...")
    size = startAtBar + 2
    logging.debug(f"Window size for kernel calculation: {size}")

    windows = [src[i:i + size].values for i in range(0, len(src) - size + 1)]
    logging.debug(f"Number of windows created: {len(windows)}")

    weight = [math.exp(-(i ** 2) / (2 * lookback ** 2)) for i in range(size)]
    logging.debug("Gaussian weights calculated for each window.")

    current_weight = [np.sum(windows[i][::-1] * weight) for i in range(0, len(src) - size + 1)]
    logging.debug("Current weight computed for each window.")

    cumulative_weight = [np.sum(weight) for _ in range(0, len(src) - size + 1)]
    logging.debug("Cumulative weight calculated.")

    gaussian_line = np.array(current_weight) / np.array(cumulative_weight)
    gaussian_line = np.concatenate((np.array([0.0] * (size - 1)), gaussian_line))
    gaussian_line = pd.Series(gaussian_line.flatten())

    logging.info("Gaussian kernel estimation completed.")
    return gaussian_line

# Types from Types.py

class __Config__:
    """
    A base configuration class for handling keyword arguments.
    It dynamically sets attributes based on the provided keyword arguments.
    """
    def __init__(self, **kwargs):
        while kwargs:
            k, v = kwargs.popitem()
            setattr(self, k, v)  # Dynamically set attributes from kwargs

class Settings(__Config__):
    """
    Settings class for configuring various parameters of the analysis.
    
    Attributes:
    source (pd.Series): Source of the input data for analysis.
    neighborsCount (int): Number of neighbors to consider in calculations.
    maxBarsBack (int): Maximum number of bars to look back for calculations.
    useDynamicExits (bool): Whether to dynamically adjust exit thresholds based on kernel regression.
    useEmaFilter (bool): Whether to apply an EMA filter.
    emaPeriod (int): Period for the Exponential Moving Average.
    useSmaFilter (bool): Whether to apply an SMA filter.
    smaPeriod (int): Period for the Simple Moving Average.
    """
    source: pd.Series
    neighborsCount = 8
    maxBarsBack = 2000
    useDynamicExits = False
    useEmaFilter = False
    emaPeriod = 200
    useSmaFilter = False
    smaPeriod = 200

class Feature:
    """
    Represents a feature used in machine learning models.
    
    Attributes:
    type (str): The type of feature (e.g., 'RSI', 'CCI').
    param1, param2 (int): Parameters used for calculating the feature.
    """
    def __init__(self, type, param1, param2):
        self.type = type
        self.param1 = param1
        self.param2 = param2

class KernelFilter(__Config__):
    """
    Configuration for kernel filtering in analysis.
    
    Attributes:
    useKernelSmoothing (bool): Flag to enable kernel smoothing.
    lookbackWindow (int): Number of bars for kernel estimation lookback.
    relativeWeight (float): Weighting factor for kernel estimation.
    regressionLevel (int): Level at which to start regression analysis.
    crossoverLag (int): Lag for crossover detection in kernel smoothing.
    """
    useKernelSmoothing = False
    lookbackWindow = 8
    relativeWeight = 8.0
    regressionLevel = 25
    crossoverLag = 2

class FilterSettings(__Config__):
    """
    Configuration for various filters used in the analysis.
    
    Attributes:
    useVolatilityFilter (bool): Whether to apply a volatility filter.
    useRegimeFilter (bool): Whether to apply a regime filter.
    useAdxFilter (bool): Whether to apply an ADX filter.
    regimeThreshold (float): Threshold for regime detection.
    adxThreshold (float): Threshold for ADX filter.
    kernelFilter (KernelFilter): Kernel filter configuration.
    """
    useVolatilityFilter = False
    useRegimeFilter = False
    useAdxFilter = False
    regimeThreshold = 0.0
    adxThreshold = 0
    kernelFilter: KernelFilter

class Filter(__Config__):
    """
    Represents the active filters in the analysis.
    
    Attributes:
    volatility (bool): Whether a volatility filter is active.
    regime (bool): Whether a regime filter is active.
    adx (bool): Whether an ADX filter is active.
    """
    volatility = False
    regime = False
    adx = False

class Direction(IntEnum):
    """
    Enumeration for representing trading directions.
    
    Attributes:
    LONG (int): Represents a long position.
    SHORT (int): Represents a short position.
    NEUTRAL (int): Represents a neutral position.
    """
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


# Classifier from Classifier.py
class LorentzianClassification:    
    """
    Class to perform Lorentzian Classification on financial data.
    
    Attributes:
    df (pd.DataFrame): The DataFrame containing financial data.
    features (list): A list of features used in the classification model.
    settings (Settings): Configuration settings for the classification.
    filterSettings (FilterSettings): Configuration for various filters.
    filter (Filter): Applied filters for the ML predictions.
    yhat1, yhat2 (pd.Series): Series used for kernel regression estimates.
    """
    df: pd.DataFrame = None
    features = list[pd.Series]()
    settings: Settings
    filterSettings: FilterSettings
    # Filter object for filtering the ML predictions
    filter: Filter

    yhat1: pd.Series
    yhat2: pd.Series

    def series_from(self, data: pd.DataFrame, feature_string, f_paramA, f_paramB) -> pd.Series:
        """
        Generates a series from the data based on the specified feature and parameters.
        
        Args:
        data (pd.DataFrame): The input data.
        feature_string (str): The feature type ('RSI', 'WT', etc.).
        f_paramA, f_paramB (int): Parameters for the feature calculation.
        
        Returns:
        pd.Series: The calculated feature series.
        """
        try:
            logging.debug("Generating series for feature: %s with params %d, %d", feature_string, f_paramA, f_paramB)
            match feature_string:
                case "RSI":
                    logging.debug("Calculating RSI")
                    return n_rsi(data['Close'], f_paramA, f_paramB)
                case "WT":
                    logging.debug("Calculating WT")
                    hlc3 = (data['High'] + data['Low'] + data['Close']) / 3
                    return n_wt(hlc3, f_paramA, f_paramB)
                case "CCI":
                    logging.debug("Calculating CCI")
                    return n_cci(data['High'], data['Low'], data['Close'], f_paramA, f_paramB)
                case "ADX":
                    logging.debug("Calculating ADX")
                    return n_adx(data['High'], data['Low'], data['Close'], f_paramA)
        except Exception as e:
            logging.error("Error in series_from for feature '%s' with params %d, %d: %s", feature_string, f_paramA, f_paramB, e)
            raise


    def __init__(self, data: pd.DataFrame, features: list = None, settings: Settings = None, filterSettings: FilterSettings = None):
        """
        Initializes the LorentzianClassification with data, features, settings, and filter settings.
        Args:
        data (pd.DataFrame): The input financial data.
        features (list): Optional. Custom features for the model.
        settings (Settings): Optional. Custom settings for the model.
        filterSettings (FilterSettings): Optional. Custom filter settings.
        """
        try:
            logging.debug("Initializing LorentzianClassification")
            self.initialize_data_features(data, features)
            logging.debug("Data features initialized")
            self.configure_settings_filters(settings, filterSettings)
            logging.debug("Settings and filters configured")
            self.calculate_kernel_regression(filterSettings)
            logging.debug("Kernel regression calculated")
            self.generate_signals_define_strategy()
            logging.debug("Signals and strategy generated")
            self.setup_entry_exit_conditions()
            logging.debug("Entry and exit conditions set up")
            
        except Exception as e:
            logging.error("Error during initialization of LorentzianClassification: %s", e)
            traceback.print_exc()
            raise
            # Calculate 'hlc3' if it doesn't exist in the DataFrame
    def initialize_data_features(self, data, features):
        logging.debug("Initializing data features in LorentzianClassification")
        if 'hlc3' not in data.columns:
            logging.debug("Adding 'hlc3' column to data")
            data['hlc3'] = (data['High'] + data['Low'] + data['Close']) / 3

        self.df = data  # No need to copy unless you need to preserve the original data
        if features is not None:
            self.features = features
            logging.debug(f"Custom features provided: {features}")
        else:
            self.features = [
                Feature("RSI", 14, 2),  # f1
                Feature("WT", 10, 11),  # f2
                Feature("CCI", 20, 2),  # f3
                Feature("ADX", 20, 2),  # f4
                Feature("RSI", 9, 2)    # f5
            ]
            logging.debug("Default features set: RSI, WT, CCI, ADX, RSI")

        logging.debug(f"Data features initialized with {len(self.features)} features")
    def configure_settings_filters(self, settings, filterSettings):
         logging.debug("Configuring settings and filters in LorentzianClassification")
 
         if settings is not None:
             self.settings = settings
             logging.debug(f"Custom settings provided: {settings}")
         else:
             self.settings = Settings(source=self.df['Close'])
             logging.debug("Default settings set with source as df['Close']")
 
         if filterSettings is not None:
             self.filterSettings = filterSettings
             logging.debug(f"Custom filter settings provided: {filterSettings}")
         else:
             self.filterSettings = FilterSettings(
                 useVolatilityFilter=True,
                 useRegimeFilter=True,
                 useAdxFilter=False,
                 regimeThreshold=-0.1,
                 adxThreshold=20,
                 kernelFilter=KernelFilter()
             )
             logging.debug("Default filter settings set")
 
         # Instantiate filters using the new 'hlc3' column
         logging.debug("Instantiating filters")
         self.filter = Filter(
             volatility=filter_volatility(self.df['High'], self.df['Low'], self.df['Close'], self.filterSettings.useVolatilityFilter, 1, 10),
             regime=regime_filter(self.df['hlc3'], self.df['High'], self.df['Low'], self.filterSettings.useRegimeFilter, self.filterSettings.regimeThreshold),
             adx=filter_adx(self.df['Close'], self.df['High'], self.df['Low'], self.filterSettings.adxThreshold, self.filterSettings.useAdxFilter, 14)
         )
         logging.debug("Filters instantiated")
 
         logging.debug(f"Configuration complete with settings: {self.settings} and filter settings: {self.filterSettings}")

    # Initialize yhat1 and yhat2 with the correct index
    def calculate_kernel_regression(self, filterSettings):
        logging.debug("Calculating kernel regression in LorentzianClassification")
        self.yhat1 = pd.Series(dtype=float, index=self.df.index)
        self.yhat2 = pd.Series(dtype=float, index=self.df.index)
        logging.debug("Initialized yhat1 and yhat2 series")

        # Generate feature series and store them in a separate list
        feature_series_list = []
        for f in self.features:
            series = self.series_from(self.df, f.type, f.param1, f.param2)
            feature_series_list.append(series)
            logging.debug(f"Added feature series: {f.type} to feature_series_list")

        # Now assign the list to self.features
        self.features = feature_series_list
        logging.debug("Assigned generated feature series to self.features")

        self.__classify()
        logging.debug("Data classified using __classify method")
        # Kernel Regression Filters: Filters based on Nadaraya-Watson Kernel Regression using the Rational Quadratic Kernel
        # For more information on this technique refer to my other open source indicator located here:
        # https://www.tradingview.com/script/AWNvbPRM-Nadaraya-Watson-Rational-Quadratic-Kernel-Non-Repainting/
        if filterSettings is None:
            self.useKernelFilter = False
            filterSettings = FilterSettings(kernelFilter=KernelFilter())
            logging.debug("No filterSettings provided, using default KernelFilter")
        elif hasattr(filterSettings, 'kernelFilter'):
            self.useKernelFilter = True
            logging.debug("Using provided kernelFilter in filterSettings")
        else:
            self.useKernelFilter = False
            filterSettings.kernelFilter = KernelFilter()
            logging.debug("kernelFilter attribute not found in filterSettings, using default KernelFilter")

        self.filterSettings = filterSettings  # Assign the filterSettings to the class attribute
        self.kFilter = self.filterSettings.kernelFilter
        
        self.yhat1 = rationalQuadratic(
            self.settings.source,
            self.filterSettings.kernelFilter.lookbackWindow,
            self.filterSettings.kernelFilter.relativeWeight,
            self.filterSettings.kernelFilter.regressionLevel
                )
        self.yhat2 = gaussian(self.settings.source, self.filterSettings.kernelFilter.lookbackWindow - self.filterSettings.kernelFilter.crossoverLag, self.filterSettings.kernelFilter.regressionLevel)
        logging.debug("Calculated yhat1 and yhat2 using kernel regression")
        # Kernel Rates of Change
        wasBearishRate = np.where(self.yhat1.shift(2) > self.yhat1.shift(1), True, False)
        wasBullishRate = np.where(self.yhat1.shift(2) < self.yhat1.shift(1), True, False)
        isBearishRate = np.where(self.yhat1.shift(1) > self.yhat1, True, False)
        isBullishRate = np.where(self.yhat1.shift(1) < self.yhat1, True, False)
        self.isBearishChange = isBearishRate & wasBullishRate
        self.isBullishChange = isBullishRate & wasBearishRate
        # Kernel Crossovers
        isBullishCrossAlert = self.crossover(self.yhat2, self.yhat1)
        isBearishCrossAlert = self.crossunder(self.yhat2, self.yhat1)
        isBullishSmooth = (self.yhat2 >= self.yhat1)
        isBearishSmooth = (self.yhat2 <= self.yhat1)
        # Kernel Colors
        # plot(kernelEstimate, color=plotColor, linewidth=2, title="Kernel Regression Estimate")
        # Alert Variables
        self.alertBullish = np.where(self.kFilter.useKernelSmoothing, isBullishCrossAlert, self.isBullishChange)
        self.alertBearish = np.where(self.kFilter.useKernelSmoothing, isBearishCrossAlert, self.isBearishChange)
        # Bullish and Bearish Filters based on Kernel
        self.isBullish = np.where(self.useKernelFilter, np.where(self.kFilter.useKernelSmoothing, isBullishSmooth, isBullishRate), True)
        self.isBearish = np.where(self.useKernelFilter, np.where(self.kFilter.useKernelSmoothing, isBearishSmooth, isBearishRate), True)
        logging.debug("Kernel regression calculation completed")
        
    def generate_signals_define_strategy(self):
        logging.debug("Generating signals and defining strategy in LorentzianClassification")
        # Convert the generator object to a numpy array to hold all predictions
        prediction = np.array([p for p in self.__get_lorentzian_predictions()])
        logging.debug("Converted Lorentzian predictions to numpy array")
        # User Defined Filters: Used for adjusting the frequency of the ML Model's predictions
        filter_all = pd.Series(self.filter.volatility & self.filter.regime & self.filter.adx)
        logging.debug("Applied user-defined filters")
        # Filtered Signal: The model's prediction of future price movement direction with user-defined filters applied
        signal = np.where(((prediction > 0) & filter_all), Direction.LONG, np.where(((prediction < 0) & filter_all), Direction.SHORT, None))
        signal[0] = (0 if signal[0] == None else signal[0])
        for i in np.where(signal == None)[0]: signal[i] = signal[i - 1 if i >= 1 else 0]
        signal = pd.Series(signal, index=self.df.index)
        logging.debug("Generated filtered signal based on predictions and filters")
        
        # Bar-Count Filters: Represents strict filters based on a pre-defined holding period of 4 bars
        # Assuming 'signal' is a pandas Series containing the trading signals
        # Determine where the signal changes type (from buy to sell or vice versa)
        isDifferentSignalType = (signal != signal.shift(1))
        # Initialize a Series to store the number of bars held since the last flip
        barsHeld = pd.Series(0, index=signal.index)
        
        # Determine the positions where the signal changes
        signal_flips = signal != signal.shift(1)
        flip_positions = np.where(signal_flips)[0]
        
        # Fill in the barsHeld Series
        for start, end in zip(flip_positions, flip_positions[1:]):
            barsHeld.iloc[start:end] = np.arange(1, end - start + 1)
        
        # Handle the period from the last flip to the end of the Series
        if flip_positions.size > 0:
            last_flip_pos = flip_positions[-1]
            barsHeld.iloc[last_flip_pos:] = np.arange(1, len(signal) - last_flip_pos + 1)
        
        # Now create the boolean Series for four bars and less than four bars
        self.isHeldFourBars = (barsHeld == 4)
        self.isHeldLessThanFourBars = (barsHeld < 4)
        logging.debug("Processed bar-count filters")
        # Fractal Filters: Derived from relative appearances of signals in a given time series fractal/segment with a default length of 4 bars
        change = lambda ser, i: (ser.shift(i, fill_value=ser.iloc[0]) != ser.shift(i+1, fill_value=ser.iloc[0]))
        isEarlySignalFlip = (change(signal, 0) & change(signal, 1) & change(signal, 2) & change(signal, 3))
        isBuySignal = ((signal == Direction.LONG) & self.df["isEmaUptrend"] & self.df["isSmaUptrend"])
        isSellSignal = ((signal == Direction.SHORT) & self.df["isEmaDowntrend"] & self.df["isSmaDowntrend"])
        isLastSignalBuy = (signal.shift(4) == Direction.LONG) & self.df["isEmaUptrend"].shift(4) & self.df["isSmaUptrend"].shift(4)
        isLastSignalSell = (signal.shift(4) == Direction.SHORT) & self.df["isEmaDowntrend"].shift(4) & self.df["isSmaDowntrend"].shift(4)
        isNewBuySignal = (isBuySignal & isDifferentSignalType)
        isNewSellSignal = (isSellSignal & isDifferentSignalType)
        logging.debug("Processed fractal filters")
        self.df["prediction"] = prediction
        self.df["signal"] = signal
        self.df["barsHeld"] = barsHeld
        # self.df["isHeldFourBars"] = isHeldFourBars
        # self.df["isHeldLessThanFourBars"] = isHeldLessThanFourBars
        self.df["isEarlySignalFlip"] = isEarlySignalFlip
        # self.df["isBuySignal"] = isBuySignal
        # self.df["isSellSignal"] = isSellSignal
        self.df["isLastSignalBuy"] = isLastSignalBuy
        self.df["isLastSignalSell"] = isLastSignalSell
        self.df["isNewBuySignal"] = isNewBuySignal
        self.df["isNewSellSignal"] = isNewSellSignal
        logging.debug("Updated dataframe with new signal and filter columns")
        
    def setup_entry_exit_conditions(self):
        logging.debug("Setting up entry and exit conditions in LorentzianClassification")
        # Entry Conditions
        startLongTrade = self.df["isNewBuySignal"] & self.isBullish & self.df["isEmaUptrend"] & self.df["isSmaUptrend"]
        startShortTrade = self.df["isNewSellSignal"] & self.isBearish & self.df["isEmaDowntrend"] & self.df["isSmaDowntrend"]
        self.df["startLongTrade"] = np.where(startLongTrade, self.df['Low'], np.NaN)
        self.df["startShortTrade"] = np.where(startShortTrade, self.df['High'], np.NaN)
        logging.debug("Entry conditions established for long and short trades")

        # Dynamic Exit Conditions
        barsSinceRedEntry = self.barssince(startShortTrade)
        barsSinceRedExit = self.barssince(self.alertBullish)
        barsSinceGreenEntry = self.barssince(startLongTrade)
        barsSinceGreenExit = self.barssince(self.alertBearish)
        isValidShortExit = barsSinceRedExit > barsSinceRedEntry
        isValidLongExit = barsSinceGreenExit > barsSinceGreenEntry
        endLongTradeDynamic = self.isBearishChange & pd.Series(isValidLongExit).shift(1)
        endShortTradeDynamic = self.isBullishChange & pd.Series(isValidShortExit).shift(1)
        logging.debug("Dynamic exit conditions processed")
        # Fixed Exit Conditions
        endLongTradeStrict = ((self.isHeldFourBars & self.df["isLastSignalBuy"]) | 
                              (self.isHeldLessThanFourBars & self.df["isNewSellSignal"] & self.df["isLastSignalBuy"])) & startLongTrade.shift(4)
        endShortTradeStrict = ((self.isHeldFourBars & self.df["isLastSignalSell"]) | 
                               (self.isHeldLessThanFourBars & self.df["isNewBuySignal"] & self.df["isLastSignalSell"])) & startShortTrade.shift(4)
        isDynamicExitValid = ~self.settings.useEmaFilter & ~self.settings.useSmaFilter & ~self.kFilter.useKernelSmoothing
        self.df["endLongTrade"] = self.settings.useDynamicExits & isDynamicExitValid & endLongTradeDynamic | endLongTradeStrict
        self.df["endShortTrade"] = self.settings.useDynamicExits & isDynamicExitValid & endShortTradeDynamic | endShortTradeStrict
        logging.debug("Fixed exit conditions established")

        logging.debug("Entry and exit conditions setup complete")
        
    def __classify(self):
        """
    Private method to classify the data using Exponential Moving Average (EMA) and 
    Simple Moving Average (SMA) indicators to determine uptrends and downtrends.

    This method updates the DataFrame `df` with new columns indicating whether each
    data point is in an uptrend or downtrend based on EMA and SMA.
    """
        try:
            logging.debug("Starting EMA and SMA classification")
            # Checks if EMA filter is to be used and calculates the uptrend and downtrend
            # for each data point based on EMA.
            self.df["isEmaUptrend"] = (self.df["Close"] > custom_ema(self.df["Close"], self.settings.emaPeriod)) if self.settings.useEmaFilter else True
            self.df["isEmaDowntrend"] = (self.df["Close"] < custom_ema(self.df["Close"], self.settings.emaPeriod)) if self.settings.useEmaFilter else True
            logging.debug("EMA classification completed")
        except Exception as e:
            print(f"An error occurred during EMA calculation: {e}")
            traceback.print_exc()
            
            # Similar to EMA, checks if SMA filter is to be used and calculates the uptrend 
            # and downtrend for each data point based on SMA.
        try:
            self.df["isSmaUptrend"] = (self.df["Close"] > custom_sma(self.df["Close"], self.settings.smaPeriod)) if self.settings.useSmaFilter else True
            self.df["isSmaDowntrend"] = (self.df["Close"] < custom_sma(self.df["Close"], self.settings.smaPeriod)) if self.settings.useSmaFilter else True
            logging.debug("SMA classification completed")
        except Exception as e:
            logging.error("Error during EMA/SMA classification: %s", e)
            traceback.print_exc()
            # 'src' refers to the source data series used for further analysis.
            src = self.settings.source

    class Distances(object):
        """
        A class to compute distances using a batch approach for efficiency.

        Attributes:
        batchSize (int): The number of items to process in each batch.
        lastBatch (int): Index of the last processed batch.
        size (int): Size of the data to process.
        features (list): List of feature series used in distance calculation.
        maxBarsBackIndex (int): Index offset for maximum bars to look back.
        dists (np.array): Array to store calculated distances for each batch.
        rows (np.array): Array to store individual rows for distance calculation.
        """
        batchSize = 50
        lastBatch = 0

        def __init__(self, features, src,maxBarsBackIndex ):
            """
            Initializes the Distances object with given features.

            Args:
            features (list): List of feature series for distance computation.
            """
            try:
                logging.debug("Initializing Distances object")
                self.size = (len(src) - maxBarsBackIndex)
                self.features = features
                self.maxBarsBackIndex = maxBarsBackIndex
                self.dists = np.array([[0.0] * self.size] * self.batchSize)
                self.rows = np.array([0.0] * self.batchSize)
                self.isBullish = None
                self.isBearish = None
                logging.debug("Distances object initialized successfully")
            except Exception as e:
                logging.error("Error during initialization of Distances object: %s", e)
                raise

        def __getitem__(self, item):
            """
            Returns the distances for a given item (index) in the dataset, calculated in batches.

            Args:
            item (int): Index of the data point for which to compute distances.

            Returns:
            np.array: Distances for the given item.
            """
            try:
                logging.debug("Getting item %s", item)
                batch = math.ceil((item + 1) / self.batchSize) * self.batchSize
                if batch > self.lastBatch:
                    logging.debug("Calculating distances for batch %s", batch)
                    self.dists.fill(0.0)
                    for feature in self.features:
                        self.rows.fill(0.0)
                        fBatch = feature[(self.maxBarsBackIndex + self.lastBatch):(self.maxBarsBackIndex + batch)]
                        self.rows[:fBatch.size] = fBatch.values
                        val = np.log(1 + cdist(pd.DataFrame(self.rows), pd.DataFrame(feature[:self.size])))
                        self.dists += val
                    self.lastBatch = batch
                    logging.debug("Distances calculated for batch %s", batch)

                return self.dists[item % self.batchSize]
            except Exception as e:
                logging.error("Error in Distances.__getitem__ for item %s: %s", item, e)
                raise
        
    def __get_lorentzian_predictions(self):
        """
        Private generator method to compute Lorentzian predictions over the dataset.

        This method yields a stream of predictions based on the Lorentzian model applied to 
        the historical data points.

        Yields:
        int: Prediction for each bar in the dataset. Zero for bars within the maxBarsBackIndex,
        otherwise a prediction based on Lorentzian classification.
        """
        try:
            # Calculate the index from which to start generating predictions
            logging.debug("Starting Lorentzian predictions generation")
            maxBarsBackIndex = (len(self.df.index) - self.settings.maxBarsBack) if (len(self.df.index) >= self.settings.maxBarsBack) else 0
            logging.debug("maxBarsBackIndex calculated: %s", maxBarsBackIndex)
            dists = self.Distances(self.features, self.df['Close'], maxBarsBackIndex)
            # Yield zero for bars within the maxBarsBackIndex
            for bar_index in range(maxBarsBackIndex):
                yield 0

            # Initialize variables for predictions and distances
            predictions = []
            distances = []

            # Get the source data series from settings
            src = self.settings.source

            # Determine the direction for training based on historical data
            y_train_array = np.where(src.shift(4) < src.shift(0), Direction.SHORT,
                                     np.where(src.shift(4) > src.shift(0), Direction.LONG, Direction.NEUTRAL))

            # Instantiate the Distances class with the features
            dists = LorentzianClassification.Distances(self.features, self.df['Close'], maxBarsBackIndex)

            # Iterate over the dataset, generating predictions
            for bar_index in range(maxBarsBackIndex, len(self.df['Close'])):
                try:
                    lastDistance = -1.0  # Initialize the last distance to a negative value
                    span = min(self.settings.maxBarsBack, bar_index + 1)  # Calculate the span
                    logging.debug("Generating predictions for bar_index: %s", bar_index)
                    # Enumerate through the distances
                    for i, d in enumerate(dists[bar_index - maxBarsBackIndex][:span]):
                        if d >= lastDistance and i % 4:
                            lastDistance = d
                            distances.append(d)
                            predictions.append(round(y_train_array[i]))
                    
                    # Adjust the list of predictions if it exceeds the neighbors count
                            if len(predictions) > self.settings.neighborsCount:
                                lastDistance = distances[round(self.settings.neighborsCount * 3 / 4)]
                                distances.pop(0)
                                predictions.pop(0)

                    # Yield the sum of predictions for the current bar_index
                                  
                    yield sum(predictions)
                except Exception as e:
                    logging.error("Error during predictions generation at bar_index %s: %s", bar_index, e)
                    traceback.print_exc()
        except Exception as e:
            logging.error("Error in __get_lorentzian_predictions method: %s", e)
            raise                
            
    def crossover(self, s1, s2):
        try:
            logging.debug("Calculating crossover between two series")
            result = (s1 > s2) & (s1.shift(1) < s2.shift(1))
            logging.debug("Crossover calculation completed")
            return result
        except Exception as e:
            logging.error("Error in crossover: %s", e)
            raise

    def crossunder(self, s1, s2):
        try:
            logging.debug("Calculating crossunder between two series")
            result = (s1 < s2) & (s1.shift(1) > s2.shift(1))
            logging.debug("Crossunder calculation completed")
            return result
        except Exception as e:
            logging.error("Error in crossunder: %s", e)
            raise

    def barssince(self, s):
        try:
            logging.debug("Calculating bars since condition met")
            s = pd.Series(s)
            val = pd.Series(0, index=s.index)
            c = np.nan
            for i in range(len(s)):
                if s.iloc[i]:
                    c = 0
                if not np.isnan(c):
                    c += 1
                    val.iloc[i] = c
            logging.debug("Bars since calculation completed")
            return val.values
        except Exception as e:
            logging.error("Error in barssince: %s", e)
            raise

    @property
    def data(self) -> pd.DataFrame:
        return self.df

    def plot(self):
        try:
            print("Plotting started")  # Print statement to confirm function call
            logging.info("Starting plot generation")
            length = len(self.df.index)  # You can adjust this to a smaller range if needed
    
            # Prepare long and short trades for plotting
            length = len(self.df)
    
            # Define subplots for kernel regression estimates and trade markers
            sub_plots = [
                mpf.make_addplot(self.df['startLongTrade'], type='scatter', markersize=50, marker='^', color='green'),
                mpf.make_addplot(self.df['startShortTrade'], type='scatter', markersize=50, marker='v', color='red'),
                ]
    
            # Define style
            s = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'figure.facecolor': 'lightgray'})
    
            # Generate plot
            fig, axlist = mpf.plot(self.df[['Open', 'High', 'Low', 'Close']], type='candle', style=s,
                                   addplot=sub_plots, figsize=(10, 6), returnfig=True)
    
            print("Plotting completed")
            plt.show()
        except Exception as e:
            print("Error in plot generation:", e)
            raise

    

spy_daily_data = yf.download('TSLA', period="5y")
print(spy_daily_data)
classifier = LorentzianClassification(data=spy_daily_data)
classifier.plot()
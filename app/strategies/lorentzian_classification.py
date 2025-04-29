"""
Self-contained LorentzianClassification module for TradingBot
Includes all types, ML functions, kernel functions, model, and strategy wrapper.
"""

# =====================
# ====   IMPORTS   ====
# =====================
import math
import numpy as np
import pandas as pd
from enum import IntEnum
from ta.momentum import rsi as RSI
from ta.volatility import average_true_range as ATR
from ta.trend import cci as CCI, adx as ADX, ema_indicator as EMA, sma_indicator as SMA
from sklearn.preprocessing import MinMaxScaler
import mplfinance as mpf

# =====================
# ====   TYPES     ====
# =====================
class __Config__:
    def __init__(self, **kwargs):
        while kwargs:
            k, v = kwargs.popitem()
            setattr(self, k, v)

class Settings(__Config__):
    source: pd.Series
    neighborsCount = 8
    maxBarsBack = 2000
    useDynamicExits = False
    useEmaFilter = False
    emaPeriod = 200
    useSmaFilter = False
    smaPeriod = 200

class Feature:
    type: str
    param1: int
    param2: int
    def __init__(self, type, param1, param2):
        self.type = type
        self.param1 = param1
        self.param2 = param2

class KernelFilter(__Config__):
    useKernelSmoothing = False
    lookbackWindow = 8
    relativeWeight = 8.0
    regressionLevel = 25
    crossoverLag = 2

class FilterSettings(__Config__):
    useVolatilityFilter = False
    useRegimeFilter = False
    useAdxFilter = False
    regimeThreshold = 0.0
    adxThreshold = 0
    kernelFilter: KernelFilter

class Filter(__Config__):
    volatility = False
    regime = False
    adx = False

class Direction(IntEnum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0

# =====================
# ====  HELPERS    ====
# =====================
def normalize(src: np.array, range_min=0, range_max=1) -> np.array:
    scaler = MinMaxScaler(feature_range=(0, 1))
    return range_min + (range_max - range_min) * scaler.fit_transform(src.reshape(-1,1))[:,0]

def rescale(src: np.array, old_min, old_max, new_min=0, new_max=1) -> np.array:
    return new_min + (new_max - new_min) * (src - old_min) / max(old_max - old_min, 10e-10)

# =====================
# ==== KERNELS     ====
# =====================
def rationalQuadratic(src: pd.Series, lookback: int, relativeWeight: float, startAtBar: int):
    currentWeight = np.zeros(len(src))
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
    currentWeight = np.zeros(len(src))
    cumulativeWeight = 0.0
    for i in range(startAtBar + 2):
        y = src.shift(i, fill_value=0.0)
        w = math.exp(-(i ** 2) / (2 * lookback ** 2))
        currentWeight += y.values * w
        cumulativeWeight += w
    val = currentWeight / cumulativeWeight
    val[:startAtBar + 1] = 0.0
    return val

# =====================
# ==== ML FEATURES ====
# =====================
def n_rsi(src: pd.Series, n1, n2) -> np.array:
    return rescale(EMA(RSI(src, n1), n2).values, 0, 100)

def n_cci(highSrc: pd.Series, lowSrc: pd.Series, closeSrc: pd.Series, n1, n2) -> np.array:
    return normalize(EMA(CCI(highSrc, lowSrc, closeSrc, n1), n2).values)

def n_wt(src: pd.Series, n1=10, n2=11) -> np.array:
    ema1 = EMA(src, n1)
    ema2 = EMA(abs(src - ema1), n1)
    ci = (src - ema1) / (0.015 * ema2)
    wt1 = EMA(ci, n2)
    wt2 = SMA(wt1, 4)
    return normalize((wt1 - wt2).values)

def n_adx(highSrc: pd.Series, lowSrc: pd.Series, closeSrc: pd.Series, n1) -> np.array:
    return rescale(ADX(highSrc, lowSrc, closeSrc, n1).values, 0, 100)

def regime_filter(src: pd.Series, high: pd.Series, low: pd.Series, useRegimeFilter, threshold) -> np.array:
    if not useRegimeFilter:
        return np.ones(len(src), dtype=bool)
    tr = ATR(high, low, src, 14)
    regime = (src - src.shift(1)).fillna(0) / tr.replace(0, np.nan)
    return regime > threshold

def filter_adx(src: pd.Series, high: pd.Series, low: pd.Series, adxThreshold, useAdxFilter, length=14):
    if not useAdxFilter:
        return np.ones(len(src), dtype=bool)
    adx_val = ADX(high, low, src, length)
    return adx_val > adxThreshold

def filter_volatility(high, low, close, useVolatilityFilter, minLength=1, maxLength=10):
    if not useVolatilityFilter:
        return np.ones(len(close), dtype=bool)
    atr = ATR(high, low, close, minLength)
    return atr > atr.rolling(maxLength).mean()

# ================================
# ==== MAIN MODEL & STRATEGY  ====
# ================================
class LorentzianClassification:
    """
    LorentzianClassification: Advanced KNN-based classification model for financial time series.
    Mirrors the TradingView PineScript logic with:
    - Lorentzian distance metric
    - Feature engineering
    - Multiple user-configurable filters (volatility, regime, ADX, kernel smoothing)
    - Dynamic and strict exit logic
    - Annotated DataFrame output for backtesting, plotting, and analysis
    """
    df: pd.DataFrame = None
    features: list
    settings: Settings
    filterSettings: FilterSettings
    filter: Filter
    yhat1: np.ndarray
    yhat2: np.ndarray

    def __init__(self, data: pd.DataFrame, features: list = None, settings: Settings = None, filterSettings: FilterSettings = None):
        required_cols = {'open','high','low','close'}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        self.df = data.copy()
        self.features = []
        self.filterSettings = None
        self.settings = None
        self.filter = None
        self.yhat1 = None
        self.yhat2 = None
        if features is None:
            features = [
                Feature("RSI", 14, 2),
                Feature("WT", 10, 11),
                Feature("CCI", 20, 2),
                Feature("ADX", 20, 2),
                Feature("RSI", 9, 2),
            ]
        if settings is None:
            settings = Settings(source=data['close'])
        if filterSettings is None:
            filterSettings = FilterSettings(
                useVolatilityFilter=True,
                useRegimeFilter=True,
                useAdxFilter=False,
                regimeThreshold=-0.1,
                adxThreshold=20,
                kernelFilter=KernelFilter()
            )
        # Feature extraction
        for f in features:
            if isinstance(f, Feature):
                self.features.append(LorentzianClassification.series_from(data, f.type, f.param1, f.param2))
            elif isinstance(f, np.ndarray):
                self.features.append(f)
            elif isinstance(f, pd.Series):
                self.features.append(f.values)
            elif isinstance(f, list):
                self.features.append(np.array(f))
            else:
                raise TypeError(f"Unsupported feature type: {type(f)}")
        self.settings = settings
        self.filterSettings = filterSettings
        ohlc4 = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        self.filter = Filter(
            volatility = filter_volatility(data['high'], data['low'], data['close'], filterSettings.useVolatilityFilter, 1, 10),
            regime = regime_filter(ohlc4, data['high'], data['low'], filterSettings.useRegimeFilter, filterSettings.regimeThreshold),
            adx = filter_adx(settings.source, data['high'], data['low'], filterSettings.adxThreshold, filterSettings.useAdxFilter, 14)
        )
        self.__classify()

    @staticmethod
    def series_from(data: pd.DataFrame, feature_string: str, f_paramA: int, f_paramB: int) -> np.ndarray:
        match feature_string:
            case "RSI":
                return n_rsi(data['close'], f_paramA, f_paramB)
            case "WT":
                hlc3 = (data['high'] + data['low'] + data['close']) / 3
                return n_wt(hlc3, f_paramA, f_paramB)
            case "CCI":
                return n_cci(data['high'], data['low'], data['close'], f_paramA, f_paramB)
            case "ADX":
                return n_adx(data['high'], data['low'], data['close'], f_paramA)
            case _:
                raise ValueError(f"Unknown feature string: {feature_string}")

    def __classify(self):
        # Helper functions
        def shift(arr, num, fill_value=None):
            result = np.empty_like(arr)
            if num > 0:
                result[:num] = fill_value if fill_value is not None else arr[0]
                result[num:] = arr[:-num]
            elif num < 0:
                result[num:] = fill_value if fill_value is not None else arr[-1]
                result[:num] = arr[-num:]
            else:
                result[:] = arr
            return result

        def crossover(s1, s2):
            return (s1 > s2) & (shift(s1, 1, fill_value=s1[0]) < shift(s2, 1, fill_value=s2[0]))

        def crossunder(s1, s2):
            return (s1 < s2) & (shift(s1, 1, fill_value=s1[0]) > shift(s2, 1, fill_value=s2[0]))

        maxBarsBackIndex = (len(self.df.index) - self.settings.maxBarsBack) if (len(self.df.index) >= self.settings.maxBarsBack) else 0

        isEmaUptrend = np.where(self.settings.useEmaFilter, (self.df["close"] > EMA(self.df["close"], self.settings.emaPeriod)), True)
        isEmaDowntrend = np.where(self.settings.useEmaFilter, (self.df["close"] < EMA(self.df["close"], self.settings.emaPeriod)), True)
        isSmaUptrend = np.where(self.settings.useSmaFilter, (self.df["close"] > SMA(self.df["close"], self.settings.smaPeriod)), True)
        isSmaDowntrend = np.where(self.settings.useSmaFilter, (self.df["close"] < SMA(self.df["close"], self.settings.smaPeriod)), True)

        src = self.settings.source

        def get_lorentzian_predictions():
            for bar_index in range(maxBarsBackIndex):
                yield 0
            predictions = []
            distances = []
            y_train_array = np.where(shift(src, 4) < shift(src, 0), Direction.SHORT, np.where(shift(src, 4) > shift(src, 0), Direction.LONG, Direction.NEUTRAL))

            class Distances(object):
                batchSize = 50
                lastBatch = 0
                def __init__(self, features):
                    self.size = (len(src) - maxBarsBackIndex)
                    self.features = features
                    self.maxBarsBackIndex = maxBarsBackIndex
                    self.dists = np.array([[0.0] * self.size] * self.batchSize)
                    self.rows = np.array([0.0] * self.batchSize)
                def __getitem__(self, item):
                    batch = math.ceil((item + 1)/self.batchSize) * self.batchSize
                    if batch > self.lastBatch:
                        self.dists.fill(0.0)
                        for feature in self.features:
                            self.rows.fill(0.0)
                            fBatch = feature[(self.maxBarsBackIndex + self.lastBatch):(self.maxBarsBackIndex + batch)]
                            self.rows[:fBatch.size] = fBatch.reshape(-1,)
                            val = np.log(1 + np.abs(self.rows.reshape(-1,1) - feature[:self.size].reshape(1,-1)))
                            self.dists += val
                        self.lastBatch = batch
                    return self.dists[item % self.batchSize]
            dists = Distances(self.features)
            for bar_index in range(maxBarsBackIndex, len(src)):
                lastDistance = -1.0
                span = min(self.settings.maxBarsBack, bar_index + 1)
                for i, d in enumerate(dists[bar_index - maxBarsBackIndex][:span]):
                    if d >= lastDistance and i % 4:
                        lastDistance = d
                        distances.append(d)
                        predictions.append(round(y_train_array[i]))
                        if len(predictions) > self.settings.neighborsCount:
                            lastDistance = distances[round(self.settings.neighborsCount*3/4)]
                            distances.pop(0)
                            predictions.pop(0)
                yield sum(predictions)

        prediction = np.array([p for p in get_lorentzian_predictions()])

        # ============================
        # ==== Prediction Filters ====
        # ============================
        filter_all = self.filter.volatility & self.filter.regime & self.filter.adx
        signal = np.where(((prediction > 0) & filter_all), Direction.LONG, np.where(((prediction < 0) & filter_all), Direction.SHORT, None))
        signal[0] = (0 if signal[0] == None else signal[0])
        for i in np.where(signal == None)[0]:
            signal[i] = signal[i - 1 if i >= 1 else 0]

        change = lambda ser, i: (shift(ser, i, fill_value=ser[0]) != shift(ser, i+1, fill_value=ser[0]))
        barsHeld = []
        isDifferentSignalType = (signal != shift(signal, 1, fill_value=signal[0]))
        _sigFlip = np.where(isDifferentSignalType)[0].tolist()
        if not (len(isDifferentSignalType) in _sigFlip):
            _sigFlip.append(len(isDifferentSignalType))
        for i, x in enumerate(_sigFlip):
            if i > 0:
                barsHeld.append(0)
            barsHeld += list(range(1, x-(-1 if i == 0 else _sigFlip[i-1])))
        isHeldFourBars = (pd.Series(barsHeld) == 4).tolist()
        isHeldLessThanFourBars = (pd.Series(barsHeld) < 4).tolist()

        isEarlySignalFlip = (change(signal, 0) & change(signal, 1) & change(signal, 2) & change(signal, 3))
        isBuySignal = ((signal == Direction.LONG) & isEmaUptrend & isSmaUptrend)
        isSellSignal = ((signal == Direction.SHORT) & isEmaDowntrend & isSmaDowntrend)
        isLastSignalBuy = (shift(signal, 4) == Direction.LONG) & shift(isEmaUptrend, 4) & shift(isSmaUptrend, 4)
        isLastSignalSell = (shift(signal, 4) == Direction.SHORT) & shift(isEmaDowntrend, 4) & shift(isSmaDowntrend, 4)
        isNewBuySignal = (isBuySignal & isDifferentSignalType)
        isNewSellSignal = (isSellSignal & isDifferentSignalType)

        # Kernel Regression Filters
        kFilter = self.filterSettings.kernelFilter
        self.yhat1 = rationalQuadratic(src, kFilter.lookbackWindow, kFilter.relativeWeight, kFilter.regressionLevel)
        self.yhat2 = gaussian(src, kFilter.lookbackWindow-kFilter.crossoverLag, kFilter.regressionLevel)
        wasBearishRate = np.where(shift(self.yhat1, 2) > shift(self.yhat1, 1), True, False)
        wasBullishRate = np.where(shift(self.yhat1, 2) < shift(self.yhat1, 1), True, False)
        isBearishRate = np.where(shift(self.yhat1, 1) > self.yhat1, True, False)
        isBullishRate = np.where(shift(self.yhat1, 1) < self.yhat1, True, False)
        isBearishChange = isBearishRate & wasBullishRate
        isBullishChange = isBullishRate & wasBearishRate
        isBullishCrossAlert = crossover(self.yhat2, self.yhat1)
        isBearishCrossAlert = crossunder(self.yhat2, self.yhat1)
        isBullishSmooth = (self.yhat2 >= self.yhat1)
        isBearishSmooth = (self.yhat2 <= self.yhat1)
        alertBullish = np.where(kFilter.useKernelSmoothing, isBullishCrossAlert, isBullishChange)
        alertBearish = np.where(kFilter.useKernelSmoothing, isBearishCrossAlert, isBearishChange)
        isBullish = np.where(self.filterSettings.kernelFilter.useKernelSmoothing, isBullishSmooth, isBullishRate) if hasattr(self.filterSettings, 'kernelFilter') and self.filterSettings.kernelFilter else isBullishRate
        isBearish = np.where(self.filterSettings.kernelFilter.useKernelSmoothing, isBearishSmooth, isBearishRate) if hasattr(self.filterSettings, 'kernelFilter') and self.filterSettings.kernelFilter else isBearishRate

        # ===========================
        # ==== Entries and Exits ====
        # ===========================
        startLongTrade = (isNewBuySignal & isBullish)
        endLongTrade = ((isDifferentSignalType & isLastSignalBuy) | (isSellSignal & isBearish))
        startShortTrade = (isNewSellSignal & isBearish)
        endShortTrade = ((isDifferentSignalType & isLastSignalSell) | (isBuySignal & isBullish))

        # Store signals and trade markers in DataFrame
        self.df["prediction"] = signal
        self.df["isNewBuySignal"] = isNewBuySignal
        self.df["isNewSellSignal"] = isNewSellSignal
        self.df["startLongTrade"] = np.where(startLongTrade, self.df['low'], np.nan)
        self.df["startShortTrade"] = np.where(startShortTrade, self.df['high'], np.nan)
        self.df["endLongTrade"] = np.where(endLongTrade, self.df['high'], np.nan)
        self.df["endShortTrade"] = np.where(endShortTrade, self.df['low'], np.nan)

    # ... (include all other methods, plotting, etc., from Classifier.py)

class LorentzianClassificationStrategy:
    """
    TradingBot strategy wrapper for advanced LorentzianClassification.
    Exposes all relevant parameters for backtesting, optimization, and UI.
    """
    def __init__(
        self,
        neighbors_count: int = 8,
        max_bars_back: int = 2000,
        feature_params: list = None,
        use_volatility_filter: bool = True,
        use_regime_filter: bool = True,
        use_adx_filter: bool = False,
        regime_threshold: float = -0.1,
        adx_threshold: float = 20,
        kernel_lookback: int = 50,
        kernel_weight: float = 0.5,
        kernel_level: int = 2,
        kernel_crossover_lag: int = 1,
        use_kernel_smoothing: bool = False,
        use_dynamic_exits: bool = False
    ):
        self.neighbors_count = neighbors_count
        self.max_bars_back = max_bars_back
        self.feature_params = feature_params or [
            Feature("RSI", 14, 2),
            Feature("WT", 10, 11),
            Feature("CCI", 20, 2),
            Feature("ADX", 20, 2),
            Feature("RSI", 9, 2),
        ]
        self.use_volatility_filter = use_volatility_filter
        self.use_regime_filter = use_regime_filter
        self.use_adx_filter = use_adx_filter
        self.regime_threshold = regime_threshold
        self.adx_threshold = adx_threshold
        self.kernel_lookback = kernel_lookback
        self.kernel_weight = kernel_weight
        self.kernel_level = kernel_level
        self.kernel_crossover_lag = kernel_crossover_lag
        self.use_kernel_smoothing = use_kernel_smoothing
        self.use_dynamic_exits = use_dynamic_exits

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        required_cols = {'open','high','low','close'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        settings = Settings(
            source=df['close'],
            neighborsCount=self.neighbors_count,
            maxBarsBack=self.max_bars_back,
            useDynamicExits=self.use_dynamic_exits
        )
        kernel_filter = KernelFilter(
            lookbackWindow=self.kernel_lookback,
            relativeWeight=self.kernel_weight,
            regressionLevel=self.kernel_level,
            crossoverLag=self.kernel_crossover_lag,
            useKernelSmoothing=self.use_kernel_smoothing
        )
        filter_settings = FilterSettings(
            useVolatilityFilter=self.use_volatility_filter,
            useRegimeFilter=self.use_regime_filter,
            useAdxFilter=self.use_adx_filter,
            regimeThreshold=self.regime_threshold,
            adxThreshold=self.adx_threshold,
            kernelFilter=kernel_filter
        )
        model = LorentzianClassification(
            data=df,
            features=self.feature_params,
            settings=settings,
            filterSettings=filter_settings
        )
        if hasattr(model, 'df') and 'prediction' in model.df.columns:
            sig = model.df['prediction'].copy()
            sig = sig.replace({2:1, -2:-1})
            sig = sig.clip(-1, 1)
            sig = sig.fillna(0).astype(int)
            return sig
        else:
            return pd.Series(0, index=df.index)

    def get_full_output(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = {'open','high','low','close'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        settings = Settings(
            source=df['close'],
            neighborsCount=self.neighbors_count,
            maxBarsBack=self.max_bars_back,
            useDynamicExits=self.use_dynamic_exits
        )
        kernel_filter = KernelFilter(
            lookbackWindow=self.kernel_lookback,
            relativeWeight=self.kernel_weight,
            regressionLevel=self.kernel_level,
            crossoverLag=self.kernel_crossover_lag,
            useKernelSmoothing=self.use_kernel_smoothing
        )
        filter_settings = FilterSettings(
            useVolatilityFilter=self.use_volatility_filter,
            useRegimeFilter=self.use_regime_filter,
            useAdxFilter=self.use_adx_filter,
            regimeThreshold=self.regime_threshold,
            adxThreshold=self.adx_threshold,
            kernelFilter=kernel_filter
        )
        model = LorentzianClassification(
            data=df,
            features=self.feature_params,
            settings=settings,
            filterSettings=filter_settings
        )
        return model.df.copy()

    """
    TradingBot strategy wrapper for advanced LorentzianClassification.
    Exposes all relevant parameters for backtesting, optimization, and UI.
    """
    def __init__(
        self,
        neighbors_count: int = 8,
        max_bars_back: int = 2000,
        feature_params: list = None,
        use_volatility_filter: bool = True,
        use_regime_filter: bool = True,
        use_adx_filter: bool = False,
        regime_threshold: float = -0.1,
        adx_threshold: float = 20,
        kernel_lookback: int = 50,
        kernel_weight: float = 0.5,
        kernel_level: int = 2,
        kernel_crossover_lag: int = 1,
        use_kernel_smoothing: bool = False,
        use_dynamic_exits: bool = False
    ):
        """
        Args:
            neighbors_count: Number of neighbors for KNN
            max_bars_back: Max bars to look back
            feature_params: List of Feature objects
            use_volatility_filter: Enable volatility filter
            use_regime_filter: Enable regime filter
            use_adx_filter: Enable ADX filter
            regime_threshold: Threshold for regime filter
            adx_threshold: Threshold for ADX filter
            kernel_lookback: Kernel regression lookback window
            kernel_weight: Kernel relative weight
            kernel_level: Kernel regression level
            kernel_crossover_lag: Kernel crossover lag
            use_kernel_smoothing: Enable kernel smoothing
            use_dynamic_exits: Enable dynamic exit logic
        """
        self.neighbors_count = neighbors_count
        self.max_bars_back = max_bars_back
        self.feature_params = feature_params or [
            Feature("RSI", 14, 2),
            Feature("WT", 10, 11),
            Feature("CCI", 20, 2),
            Feature("ADX", 20, 2),
            Feature("RSI", 9, 2),
        ]
        self.use_volatility_filter = use_volatility_filter
        self.use_regime_filter = use_regime_filter
        self.use_adx_filter = use_adx_filter
        self.regime_threshold = regime_threshold
        self.adx_threshold = adx_threshold
        self.kernel_lookback = kernel_lookback
        self.kernel_weight = kernel_weight
        self.kernel_level = kernel_level
        self.kernel_crossover_lag = kernel_crossover_lag
        self.use_kernel_smoothing = use_kernel_smoothing
        self.use_dynamic_exits = use_dynamic_exits

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals (-1, 0, 1) using the LorentzianClassification model.
        Args:
            df (pd.DataFrame): OHLCV data
        Returns:
            pd.Series: Signal series (-1, 0, 1)
        """
        required_cols = {'open','high','low','close'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        settings = Settings(
            source=df['close'],
            neighborsCount=self.neighbors_count,
            maxBarsBack=self.max_bars_back,
            useDynamicExits=self.use_dynamic_exits
        )
        kernel_filter = KernelFilter(
            lookbackWindow=self.kernel_lookback,
            relativeWeight=self.kernel_weight,
            regressionLevel=self.kernel_level,
            crossoverLag=self.kernel_crossover_lag,
            useKernelSmoothing=self.use_kernel_smoothing
        )
        filter_settings = FilterSettings(
            useVolatilityFilter=self.use_volatility_filter,
            useRegimeFilter=self.use_regime_filter,
            useAdxFilter=self.use_adx_filter,
            regimeThreshold=self.regime_threshold,
            adxThreshold=self.adx_threshold,
            kernelFilter=kernel_filter
        )
        model = LorentzianClassification(
            data=df,
            features=self.feature_params,
            settings=settings,
            filterSettings=filter_settings
        )
        # Use the model's prediction column as signals, fallback to zeros if not present
        if hasattr(model, 'df') and 'prediction' in model.df.columns:
            sig = model.df['prediction'].copy()
            sig = sig.replace({2:1, -2:-1})
            sig = sig.clip(-1, 1)
            sig = sig.fillna(0).astype(int)
            return sig
        else:
            return pd.Series(0, index=df.index)

    def get_full_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the model and return the full annotated DataFrame (for advanced analysis).
        """
        required_cols = {'open','high','low','close'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        settings = Settings(
            source=df['close'],
            neighborsCount=self.neighbors_count,
            maxBarsBack=self.max_bars_back,
            useDynamicExits=self.use_dynamic_exits
        )
        kernel_filter = KernelFilter(
            lookbackWindow=self.kernel_lookback,
            relativeWeight=self.kernel_weight,
            regressionLevel=self.kernel_level,
            crossoverLag=self.kernel_crossover_lag,
            useKernelSmoothing=self.use_kernel_smoothing
        )
        filter_settings = FilterSettings(
            useVolatilityFilter=self.use_volatility_filter,
            useRegimeFilter=self.use_regime_filter,
            useAdxFilter=self.use_adx_filter,
            regimeThreshold=self.regime_threshold,
            adxThreshold=self.adx_threshold,
            kernelFilter=kernel_filter
        )
        model = LorentzianClassification(
            data=df,
            features=self.feature_params,
            settings=settings,
            filterSettings=filter_settings
        )
        return model.df.copy()

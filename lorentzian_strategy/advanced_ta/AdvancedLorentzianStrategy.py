"""
Advanced Lorentzian Classification Strategy
==========================================

This strategy integrates the complete LorentzianClassification package into a single
Lumibot-compatible strategy file. It includes:

1. Advanced ML-based classification using Lorentzian distance
2. Multiple technical indicators (RSI, WaveTrend, CCI, ADX)
3. Kernel regression filters for signal smoothing
4. Multiple filter types (volatility, regime, ADX)
5. Dynamic and fixed exit strategies

Based on the research by @jdehorty on TradingView and enhanced for Lumibot.
"""

import pandas as pd
import numpy as np
import math
import json
import os
from typing import List, Optional, Dict, Any
from enum import IntEnum
from dataclasses import dataclass
from dotenv import load_dotenv

from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.entities import Asset, Order, TradingFee
from lumibot.backtesting import PolygonDataBacktesting
from lumibot.credentials import IS_BACKTESTING

# Load environment variables
load_dotenv()

# Technical analysis imports
from ta.momentum import rsi as ta_rsi
from ta.volatility import average_true_range as ATR
from ta.trend import cci as ta_cci, adx as ta_adx, ema_indicator as ta_ema, sma_indicator as ta_sma
from sklearn.preprocessing import MinMaxScaler

# =====================================
# ======== CORE DATA STRUCTURES ======
# =====================================

class Direction(IntEnum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0

@dataclass
class Feature:
    type: str
    param1: int
    param2: int

@dataclass
class KernelFilter:
    useKernelSmoothing: bool = False
    lookbackWindow: int = 8
    relativeWeight: float = 8.0
    regressionLevel: int = 25
    crossoverLag: int = 2

@dataclass
class FilterSettings:
    useVolatilityFilter: bool = True
    useRegimeFilter: bool = True
    useAdxFilter: bool = False
    regimeThreshold: float = -0.1
    adxThreshold: int = 20
    kernelFilter: KernelFilter = None

    def __post_init__(self):
        if self.kernelFilter is None:
            self.kernelFilter = KernelFilter()

@dataclass
class ClassifierSettings:
    neighborsCount: int = 8
    maxBarsBack: int = 2000
    useDynamicExits: bool = False
    useEmaFilter: bool = False
    emaPeriod: int = 200
    useSmaFilter: bool = False
    smaPeriod: int = 200

@dataclass
class FilterState:
    volatility: np.ndarray
    regime: np.ndarray
    adx: np.ndarray

# =====================================
# ======== UTILITY FUNCTIONS =========
# =====================================

def shift(arr: np.ndarray, length: int, fill_value: float = 0.0) -> np.ndarray:
    """Shift array by specified length, filling with fill_value"""
    return np.pad(arr, (length,), mode='constant', constant_values=(fill_value,))[:arr.size]

def barssince(s: np.array):
    val = np.array([0.0]*s.size)
    c = math.nan
    for i in range(s.size):
        if s[i]: c = 0; continue
        if c >= 0: c += 1
        val[i] = c
    return val

def crossover(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Detect crossover: s1 crosses above s2"""
    return (s1 > s2) & (shift(s1, 1) < shift(s2, 1))

def crossunder(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Detect crossunder: s1 crosses below s2"""
    return (s1 < s2) & (shift(s1, 1) > shift(s2, 1))

# =====================================
# ======== INDICATOR FUNCTIONS =======
# =====================================

def normalize(src: np.ndarray, range_min: float = 0, range_max: float = 1) -> np.ndarray:
    """Normalize array to specified range"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized = scaler.fit_transform(src.reshape(-1, 1))[:, 0]
    return range_min + (range_max - range_min) * normalized

def rescale(src: np.ndarray, old_min: float, old_max: float, new_min: float = 0, new_max: float = 1) -> np.ndarray:
    """Rescale array from one range to another"""
    rescaled_value = new_min + (new_max - new_min) * (src - old_min) / max(old_max - old_min, 1e-10)
    return rescaled_value

def n_rsi(src: pd.Series, n1: int, n2: int) -> np.ndarray:
    """Normalized RSI for ML algorithms"""
    rsi_values = ta_rsi(src, n1)
    smoothed_rsi = ta_ema(rsi_values, n2)
    return rescale(smoothed_rsi.values, 0, 100)

def n_cci(high: pd.Series, low: pd.Series, close: pd.Series, n1: int, n2: int) -> np.ndarray:
    """Normalized CCI for ML algorithms"""
    cci_values = ta_cci(high, low, close, n1)
    smoothed_cci = ta_ema(cci_values, n2)
    return normalize(smoothed_cci.values)

def n_wt(src: pd.Series, n1: int = 10, n2: int = 11) -> np.ndarray:
    """Normalized WaveTrend for ML algorithms"""
    esa = ta_ema(src, n1)
    de = ta_ema(abs(src - esa), n1)
    ci = (src - esa) / (0.015 * de)
    wt1 = ta_ema(ci, n2)
    wt2 = ta_sma(wt1, 4)
    return normalize((wt1 - wt2).values)

def n_adx(high: pd.Series, low: pd.Series, close: pd.Series, n1: int) -> np.ndarray:
    """Normalized ADX for ML algorithms"""
    adx_values = ta_adx(high, low, close, n1)
    return rescale(adx_values.values, 0, 100)

# =====================================
# ======== KERNEL FUNCTIONS ==========
# =====================================

def rational_quadratic_kernel(src: pd.Series, lookback: int, relative_weight: float, start_at_bar: int) -> np.ndarray:
    """Rational Quadratic Kernel for regression"""
    current_weight = np.zeros(len(src))
    cumulative_weight = 0.0
    
    for i in range(start_at_bar + 2):
        y = src.shift(i, fill_value=0.0)
        w = (1 + (i ** 2 / (lookback ** 2 * 2 * relative_weight))) ** -relative_weight
        current_weight += y.values * w
        cumulative_weight += w
    
    val = current_weight / cumulative_weight
    val[:start_at_bar + 1] = 0.0
    return val

def gaussian_kernel(src: pd.Series, lookback: int, start_at_bar: int) -> np.ndarray:
    """Gaussian Kernel for regression"""
    current_weight = np.zeros(len(src))
    cumulative_weight = 0.0
    
    for i in range(start_at_bar + 2):
        y = src.shift(i, fill_value=0.0)
        w = math.exp(-(i ** 2) / (2 * lookback ** 2))
        current_weight += y.values * w
        cumulative_weight += w
    
    val = current_weight / cumulative_weight
    val[:start_at_bar + 1] = 0.0
    return val

# =====================================
# ======== FILTER FUNCTIONS ==========
# =====================================

def regime_filter(src: pd.Series, high: pd.Series, low: pd.Series, use_filter: bool, threshold: float) -> np.ndarray:
    """Regime filter for trend/range detection"""
    if not use_filter:
        return np.array([True] * len(src))
    
    def klmf(src_vals: np.ndarray, high_vals: np.ndarray, low_vals: np.ndarray):
        value1 = np.zeros(len(src_vals))
        value2 = np.zeros(len(src_vals))
        klmf_vals = np.zeros(len(src_vals))
        
        for i in range(len(src_vals)):
            if (high_vals[i] - low_vals[i]) == 0:
                continue
            value1[i] = 0.2 * (src_vals[i] - src_vals[i-1 if i >= 1 else 0]) + 0.8 * value1[i-1 if i >= 1 else 0]
            value2[i] = 0.1 * (high_vals[i] - low_vals[i]) + 0.8 * value2[i-1 if i >= 1 else 0]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            omega = np.nan_to_num(np.abs(np.divide(value1, value2)))
        alpha = (-(omega ** 2) + np.sqrt((omega ** 4) + 16 * (omega ** 2))) / 8
        
        for i in range(len(src_vals)):
            klmf_vals[i] = alpha[i] * src_vals[i] + (1 - alpha[i]) * klmf_vals[i-1 if i >= 1 else 0]
        
        return klmf_vals
    
    filter_result = np.array([False] * len(src))
    abs_curve_slope = np.abs(np.diff(klmf(src.values, high.values, low.values), prepend=0.0))
    exp_avg_abs_curve_slope = ta_ema(pd.Series(abs_curve_slope), 200).values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_slope_decline = (abs_curve_slope - exp_avg_abs_curve_slope) / exp_avg_abs_curve_slope
    
    flags = (normalized_slope_decline >= threshold)
    filter_result[(len(filter_result) - len(flags)):] = flags
    return filter_result

def filter_adx(src: pd.Series, high: pd.Series, low: pd.Series, adx_threshold: int, use_filter: bool, length: int = 14) -> np.ndarray:
    """ADX filter for trend strength"""
    if not use_filter:
        return np.array([True] * len(src))
    adx_values = ta_adx(high, low, src, length).values
    return (adx_values > adx_threshold)

def filter_volatility(high: pd.Series, low: pd.Series, close: pd.Series, use_filter: bool, min_length: int = 1, max_length: int = 10) -> np.ndarray:
    """Volatility filter"""
    if not use_filter:
        return np.array([True] * len(close))
    recent_atr = ATR(high, low, close, min_length).values
    historical_atr = ATR(high, low, close, max_length).values
    return (recent_atr > historical_atr)

# =====================================
# ======== MAIN CLASSIFIER ===========
# =====================================

class LorentzianClassifier:
    """Advanced Lorentzian Classification for ML-based trading signals"""
    
    def __init__(self, data: pd.DataFrame, features: List[Feature] = None, 
                 settings: ClassifierSettings = None, filter_settings: FilterSettings = None):
        self.df = data.copy()
        self.features = []
        self.settings = settings or ClassifierSettings()
        self.filter_settings = filter_settings or FilterSettings()
        
        # Default features if none provided
        if features is None:
            features = [
                Feature("RSI", 14, 2),
                Feature("WT", 10, 11),
                Feature("CCI", 20, 2),
                Feature("ADX", 20, 2),
                Feature("RSI", 9, 2),
            ]
        
        # Calculate feature series
        for feature in features:
            self.features.append(self._series_from(data, feature.type, feature.param1, feature.param2))
        
        # Calculate filters
        ohlc4 = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        self.filters = FilterState(
            volatility=filter_volatility(data['high'], data['low'], data['close'], 
                                       self.filter_settings.useVolatilityFilter, 1, 10),
            regime=regime_filter(ohlc4, data['high'], data['low'], 
                               self.filter_settings.useRegimeFilter, 
                               self.filter_settings.regimeThreshold),
            adx=filter_adx(data['close'], data['high'], data['low'], 
                          self.filter_settings.adxThreshold, 
                          self.filter_settings.useAdxFilter, 14)
        )
        
        # Run classification
        self._classify()
    
    def _series_from(self, data: pd.DataFrame, feature_string: str, param1: int, param2: int) -> np.ndarray:
        """Generate feature series based on feature type"""
        if feature_string == "RSI":
            return n_rsi(data['close'], param1, param2)
        elif feature_string == "WT":
            hlc3 = (data['high'] + data['low'] + data['close']) / 3
            return n_wt(hlc3, param1, param2)
        elif feature_string == "CCI":
            return n_cci(data['high'], data['low'], data['close'], param1, param2)
        elif feature_string == "ADX":
            return n_adx(data['high'], data['low'], data['close'], param1)
        else:
            raise ValueError(f"Unknown feature type: {feature_string}")
    
    def _classify(self):
        """Main classification logic using Lorentzian distance k-NN"""
        src = self.df['close']
        max_bars_back_index = max(0, len(self.df) - self.settings.maxBarsBack)
        
        # Trend filters
        ema_values = ta_ema(src, self.settings.emaPeriod) if self.settings.useEmaFilter else None
        sma_values = ta_sma(src, self.settings.smaPeriod) if self.settings.useSmaFilter else None
        
        is_ema_uptrend = (src > ema_values) if self.settings.useEmaFilter else np.array([True] * len(src))
        is_ema_downtrend = (src < ema_values) if self.settings.useEmaFilter else np.array([True] * len(src))
        is_sma_uptrend = (src > sma_values) if self.settings.useSmaFilter else np.array([True] * len(src))
        is_sma_downtrend = (src < sma_values) if self.settings.useSmaFilter else np.array([True] * len(src))
        
        # Generate predictions using Lorentzian k-NN
        predictions = self._get_lorentzian_predictions(src, max_bars_back_index)
        
        # Apply filters
        filter_all = self.filters.volatility & self.filters.regime & self.filters.adx
        
        # Generate signals
        signal = np.where(
            (predictions > 0) & filter_all, Direction.LONG,
            np.where((predictions < 0) & filter_all, Direction.SHORT, Direction.NEUTRAL)
        )
        
        # Forward fill signals
        for i in range(1, len(signal)):
            if signal[i] == Direction.NEUTRAL:
                signal[i] = signal[i-1]
        
        # Calculate entry/exit conditions
        is_different_signal_type = (signal != shift(signal, 1, fill_value=signal[0]))
        is_new_buy_signal = (signal == Direction.LONG) & is_different_signal_type & is_ema_uptrend & is_sma_uptrend
        is_new_sell_signal = (signal == Direction.SHORT) & is_different_signal_type & is_ema_downtrend & is_sma_downtrend
        
        # Apply kernel regression if enabled
        kernel_filter = self.filter_settings.kernelFilter
        if kernel_filter and hasattr(kernel_filter, 'useKernelSmoothing'):
            yhat1 = rational_quadratic_kernel(src, kernel_filter.lookbackWindow, 
                                            kernel_filter.relativeWeight, kernel_filter.regressionLevel)
            yhat2 = gaussian_kernel(src, kernel_filter.lookbackWindow - kernel_filter.crossoverLag, 
                                  kernel_filter.regressionLevel)
            
            # Kernel-based filters
            is_bullish_cross = crossover(yhat2, yhat1)
            is_bearish_cross = crossunder(yhat2, yhat1)
            is_bullish_smooth = (yhat2 >= yhat1)
            is_bearish_smooth = (yhat2 <= yhat1)
            
            is_bullish_rate = (shift(yhat1, 1) < yhat1)
            is_bearish_rate = (shift(yhat1, 1) > yhat1)
            
            if kernel_filter.useKernelSmoothing:
                is_bullish = is_bullish_smooth
                is_bearish = is_bearish_smooth
                alert_bullish = is_bullish_cross
                alert_bearish = is_bearish_cross
            else:
                is_bullish = is_bullish_rate
                is_bearish = is_bearish_rate
                alert_bullish = is_bullish_rate & shift(is_bearish_rate, 1)
                alert_bearish = is_bearish_rate & shift(is_bullish_rate, 1)
        else:
            is_bullish = np.array([True] * len(src))
            is_bearish = np.array([True] * len(src))
            alert_bullish = np.array([False] * len(src))
            alert_bearish = np.array([False] * len(src))
        
        # Final entry conditions
        start_long_trade = is_new_buy_signal & is_bullish & is_ema_uptrend & is_sma_uptrend
        start_short_trade = is_new_sell_signal & is_bearish & is_ema_downtrend & is_sma_downtrend
        
        # Store results in dataframe
        self.df['prediction'] = predictions
        self.df['signal'] = signal
        self.df['start_long'] = start_long_trade
        self.df['start_short'] = start_short_trade
        self.df['is_bullish'] = is_bullish
        self.df['is_bearish'] = is_bearish
        
        return self.df
    
    def _get_lorentzian_predictions(self, src: pd.Series, max_bars_back_index: int) -> np.ndarray:
        """Generate predictions using Lorentzian distance k-NN algorithm"""
        predictions = np.zeros(len(src))
        
        # Create training labels (next 4-bar direction)
        y_train = np.where(
            src.shift(-4) > src, Direction.LONG,
            np.where(src.shift(-4) < src, Direction.SHORT, Direction.NEUTRAL)
        )
        
        for bar_index in range(max_bars_back_index, len(src)):
            if bar_index < len(self.features[0]):  # Ensure we have feature data
                # Current feature vector
                current_features = [feature[bar_index] if bar_index < len(feature) else 0 
                                  for feature in self.features]
                
                # Skip if any current features are NaN
                if any(np.isnan(f) for f in current_features):
                    predictions[bar_index] = 0
                    continue
                
                # Calculate distances to historical points
                distances = []
                span = min(self.settings.maxBarsBack, bar_index)
                
                for i in range(max(0, bar_index - span), bar_index):
                    if i % 4 != 0:  # Skip for chronological spacing
                        continue
                    
                    # Historical feature vector
                    hist_features = [feature[i] if i < len(feature) else 0 
                                   for feature in self.features]
                    
                    # Skip if any historical features are NaN
                    if any(np.isnan(f) for f in hist_features):
                        continue
                    
                    # Calculate Lorentzian distance
                    distance = sum(math.log(1 + abs(a - b)) 
                                 for a, b in zip(current_features, hist_features))
                    distances.append((distance, y_train[i]))
                
                # Get k nearest neighbors
                if len(distances) >= self.settings.neighborsCount:
                    distances.sort(key=lambda x: x[0])
                    nearest = distances[:self.settings.neighborsCount]
                    prediction_sum = sum(label for _, label in nearest)
                    predictions[bar_index] = prediction_sum
                else:
                    predictions[bar_index] = 0
        
        return predictions

# =====================================
# ======== LUMIBOT STRATEGY ==========
# =====================================

class AdvancedLorentzianStrategy(Strategy):
    """
    Advanced Lorentzian Classification Strategy for Lumibot
    
    This strategy uses machine learning with Lorentzian distance to classify
    market conditions and generate trading signals. It includes multiple
    technical indicators, kernel regression, and various filters.
    
    By default, it will try to load optimized parameters from best_parameters_{symbol}.json
    generated by optimize_parameters.py. If the file doesn't exist, it uses class defaults.
    
    Set environment variable USE_OPTIMIZED_PARAMS=false to disable optimization.
    """
    
    parameters = {
        'symbols': [os.getenv('SYMBOL')],  # Read from .env file
        'neighbors': 8,
        'history_window': 2000,
        'max_bars_back': 500,
        'use_dynamic_exits': False,
        'use_ema_filter': False,
        'ema_period': 200,
        'use_sma_filter': False,
        'sma_period': 200,
        'use_volatility_filter': True,
        'use_regime_filter': True,
        'use_adx_filter': False,
        'regime_threshold': -0.1,
        'adx_threshold': 20,
        'use_kernel_smoothing': False,
        'kernel_lookback': 8,
        'kernel_weight': 8.0,
        'regression_level': 25,
        'crossover_lag': 2,
        'features': [
            ('RSI', 14, 2),
            ('WT', 10, 11),
            ('CCI', 20, 2),
            ('ADX', 20, 2),
            ('RSI', 9, 2)
        ]
    }
    
    def load_optimized_parameters(self, symbol):
        """Load optimized parameters from JSON file if available"""
        try:
            # Look for the file in the same directory as this script
            script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_logs")
            best_params_file = os.path.join(script_dir, f"best_parameters_{symbol}.json")
            
            print(f"🔍 Checking for optimized parameters...")
            print(f"   Looking for: {best_params_file}")
            
            if os.path.exists(best_params_file):
                print(f"✅ Found optimization file for {symbol}")
                
                with open(best_params_file, 'r') as f:
                    data = json.load(f)
                
                if 'lumibot_parameters' in data:
                    print(f"🎯 Loading optimized parameters for {symbol}")
                    print(f"   📅 Optimization date: {data['optimization_info']['optimization_date'][:10]}")
                    print(f"   📈 Expected return: {data['optimization_info']['total_return']:+.2f}%")
                    print(f"   🎯 Expected win rate: {data['optimization_info']['win_rate']:.1f}%")
                    print(f"   🔢 Total trades: {data['optimization_info']['total_trades']}")
                    print(f"   💰 Final portfolio value: ${data['optimization_info']['final_portfolio_value']:,.2f}")
                    
                    return data['lumibot_parameters']
                else:
                    print(f"⚠️  Old parameter format found for {symbol}")
                    print(f"   Please re-run optimization: python optimize_parameters.py")
                    return None
                    
            else:
                print(f"❌ No optimized parameters found for {symbol}")
                print(f"   File not found: {best_params_file}")
                print(f"   💡 Run optimization first: python optimize_parameters.py")
                return None
                
        except Exception as e:
            print(f"❌ Error loading optimized parameters: {e}")
            return None
    
    def initialize(self):
        """Initialize strategy parameters and settings"""
        self.sleeptime = "1D"  # Run once per day
        self.set_market("NYSE")
        
        # Extract parameters
        p = self.parameters.copy()
        
        # Read symbol from environment variable, with fallback to parameters
        env_symbol = os.getenv('SYMBOL')
        if env_symbol:
            self.symbols = [env_symbol]
        else:
            self.symbols = p.get('symbols')
        
        # Check environment variable (defaults to True)
        use_optimized_params = os.getenv('USE_OPTIMIZED_PARAMS', 'true').lower() == 'true'
        
        print("="*80)
        print("🎯 ADVANCED LORENTZIAN CLASSIFICATION STRATEGY")
        print("="*80)
        print(f"📊 Symbols: {self.symbols}")
        print(f"⚙️  Use optimized params: {use_optimized_params} (env: {os.getenv('USE_OPTIMIZED_PARAMS', 'default: true')})")
        print()
        
        # Load optimized parameters if requested
        if use_optimized_params:
            optimization_symbol = self.symbols[0]  # Use first symbol for optimization
            print(f"🔧 PARAMETER OPTIMIZATION CHECK FOR {optimization_symbol}")
            print("-" * 50)
            
            optimized_params = self.load_optimized_parameters(optimization_symbol)
            
            if optimized_params:
                # Override default parameters with optimized ones
                print(f"✅ SUCCESS: Using optimized parameters for {optimization_symbol}")
                p.update(optimized_params)
                param_source = "OPTIMIZED"
            else:
                print(f"⚠️  FALLBACK: Using class default parameters")
                param_source = "DEFAULT"
        else:
            print(f"📋 OPTIMIZATION DISABLED: Using class default parameters")
            param_source = "DEFAULT"
        
        print()
        
        # Create classifier settings
        self.classifier_settings = ClassifierSettings(
            neighborsCount=p.get('neighbors', 8),
            maxBarsBack=p.get('max_bars_back', 500),
            useDynamicExits=p.get('use_dynamic_exits', False),
            useEmaFilter=p.get('use_ema_filter', False),
            emaPeriod=p.get('ema_period', 200),
            useSmaFilter=p.get('use_sma_filter', False),
            smaPeriod=p.get('sma_period', 200)
        )
        
        # Create filter settings
        kernel_filter = KernelFilter(
            useKernelSmoothing=p.get('use_kernel_smoothing', False),
            lookbackWindow=p.get('kernel_lookback', 8),
            relativeWeight=p.get('kernel_weight', 8.0),
            regressionLevel=p.get('regression_level', 25),
            crossoverLag=p.get('crossover_lag', 2)
        )
        
        self.filter_settings = FilterSettings(
            useVolatilityFilter=p.get('use_volatility_filter', True),
            useRegimeFilter=p.get('use_regime_filter', True),
            useAdxFilter=p.get('use_adx_filter', False),
            regimeThreshold=p.get('regime_threshold', -0.1),
            adxThreshold=p.get('adx_threshold', 20),
            kernelFilter=kernel_filter
        )
        
        # Create features
        self.features = []
        for feature_config in p.get('features', [('RSI', 14, 2), ('WT', 10, 11), ('CCI', 20, 2)]):
            if len(feature_config) == 3:
                self.features.append(Feature(feature_config[0], feature_config[1], feature_config[2]))
        
        # Log final configuration
        print(f"⚙️  FINAL STRATEGY CONFIGURATION ({param_source} PARAMETERS)")
        print("-" * 50)
        print(f"🧠 ML Settings:")
        print(f"   • Neighbors: {self.classifier_settings.neighborsCount}")
        print(f"   • Max bars back: {self.classifier_settings.maxBarsBack}")
        print(f"   • Dynamic exits: {self.classifier_settings.useDynamicExits}")
        print(f"📊 Trend Filters:")
        print(f"   • EMA filter: {self.classifier_settings.useEmaFilter} (period: {self.classifier_settings.emaPeriod})")
        print(f"   • SMA filter: {self.classifier_settings.useSmaFilter} (period: {self.classifier_settings.smaPeriod})")
        print(f"🎛️  Advanced Filters:")
        print(f"   • Volatility: {self.filter_settings.useVolatilityFilter}")
        print(f"   • Regime: {self.filter_settings.useRegimeFilter} (threshold: {self.filter_settings.regimeThreshold})")
        print(f"   • ADX: {self.filter_settings.useAdxFilter} (threshold: {self.filter_settings.adxThreshold})")
        print(f"   • Kernel smoothing: {self.filter_settings.kernelFilter.useKernelSmoothing}")
        print(f"📈 Features: {[(f.type, f.param1, f.param2) for f in self.features]}")
        print("="*80)
        print()
        
        # Initialize tracking variables
        if not hasattr(self.vars, 'position_state'):
            self.vars.position_state = {symbol: 'neutral' for symbol in self.symbols}
        if not hasattr(self.vars, 'last_signal'):
            self.vars.last_signal = {symbol: 0 for symbol in self.symbols}
    
    def on_trading_iteration(self):
        """Main trading logic executed on each iteration"""
        for symbol in self.symbols:
            try:
                asset = Asset(symbol, Asset.AssetType.STOCK)
                
                # Get historical data
                history_length = max(self.classifier_settings.maxBarsBack, self.parameters.get('history_window', 500))
                bars = self.get_historical_prices(asset, history_length, "day")
                
                if bars is None or len(bars.df) < 100:
                    self.log_message(f"Insufficient data for {symbol}", color="yellow")
                    continue
                
                df = bars.df.copy()
                
                # Ensure we have required columns
                if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                    self.log_message(f"Missing required columns for {symbol}", color="red")
                    continue
                
                # Run Lorentzian Classification
                classifier = LorentzianClassifier(
                    data=df,
                    features=self.features,
                    settings=self.classifier_settings,
                    filter_settings=self.filter_settings
                )
                
                # Get latest signals
                if len(classifier.df) == 0:
                    self.log_message(f"No classification results for {symbol}", color="yellow")
                    continue
                
                latest_idx = len(classifier.df) - 1
                current_signal = classifier.df.iloc[latest_idx]['signal']
                start_long = classifier.df.iloc[latest_idx]['start_long']
                start_short = classifier.df.iloc[latest_idx]['start_short']
                current_price = classifier.df.iloc[latest_idx]['close']
                
                # Log signal information
                if current_signal != self.vars.last_signal.get(symbol, 0):
                    signal_name = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}.get(current_signal, "UNKNOWN")
                    self.log_message(f"{symbol}: Signal changed to {signal_name} at ${current_price:.2f}", 
                                   color="blue")
                    self.vars.last_signal[symbol] = current_signal
                
                # Get current position
                position = self.get_position(asset)
                current_qty = position.quantity if position else 0
                
                # Execute trading logic
                if start_long and current_qty <= 0:
                    # Enter long position
                    if current_qty < 0:
                        # Close short position first
                        close_order = self.create_order(asset, abs(current_qty), Order.OrderSide.BUY)
                        self.submit_order(close_order)
                        self.log_message(f"Closed SHORT position: {abs(current_qty)} {symbol} @ ${current_price:.2f}", 
                                       color="orange")
                    
                    # Open long position
                    cash = self.get_cash()
                    position_size = min(cash * 0.95, cash - 1000)  # Leave some cash buffer
                    qty = int(position_size / current_price)
                    
                    if qty > 0:
                        buy_order = self.create_order(asset, qty, Order.OrderSide.BUY)
                        self.submit_order(buy_order)
                        self.log_message(f"Opened LONG position: {qty} {symbol} @ ${current_price:.2f}", 
                                       color="green")
                        self.vars.position_state[symbol] = 'long'
                
                elif start_short and current_qty >= 0:
                    # Enter short position (if allowed)
                    if current_qty > 0:
                        # Close long position first
                        close_order = self.create_order(asset, current_qty, Order.OrderSide.SELL)
                        self.submit_order(close_order)
                        self.log_message(f"Closed LONG position: {current_qty} {symbol} @ ${current_price:.2f}", 
                                       color="orange")
                        self.vars.position_state[symbol] = 'neutral'
                    
                    # Note: Short selling may not be available in all brokers/environments
                    # Uncomment below if short selling is supported:
                    
                    # portfolio_value = self.get_portfolio_value()
                    # position_size = min(portfolio_value * 0.95, portfolio_value - 1000)
                    # qty = int(position_size / current_price)
                    # 
                    # if qty > 0:
                    #     short_order = self.create_order(asset, qty, Order.OrderSide.SELL)
                    #     self.submit_order(short_order)
                    #     self.log_message(f"Opened SHORT position: {qty} {symbol} @ ${current_price:.2f}", 
                    #                    color="red")
                    #     self.vars.position_state[symbol] = 'short'
                
                # Add charting
                self.add_line(f"{symbol}_price", current_price, color="black")
                
                # Add signal markers
                if start_long:
                    self.add_marker(f"{symbol}_long", current_price, color="green", 
                                  symbol="triangle-up", size=10)
                elif start_short:
                    self.add_marker(f"{symbol}_short", current_price, color="red", 
                                  symbol="triangle-down", size=10)
                
            except Exception as e:
                self.log_message(f"Error processing {symbol}: {str(e)}", color="red")
                continue
    
    def on_abrupt_closing(self):
        """Handle strategy shutdown gracefully"""
        self.log_message("Strategy shutting down, closing all positions...", color="yellow")
        for symbol in self.symbols:
            try:
                asset = Asset(symbol, Asset.AssetType.STOCK)
                position = self.get_position(asset)
                if position and position.quantity != 0:
                    side = Order.OrderSide.SELL if position.quantity > 0 else Order.OrderSide.BUY
                    order = self.create_order(asset, abs(position.quantity), side)
                    self.submit_order(order)
            except Exception as e:
                self.log_message(f"Error closing position for {symbol}: {str(e)}", color="red")

# =====================================
# ============= RUNNER ================
# =====================================

if __name__ == "__main__":
    # Override parameters as needed
    strategy_params = AdvancedLorentzianStrategy.parameters.copy()
    
    # Example: Use different symbols or disable optimization
    # strategy_params['symbols'] = ['AAPL', 'MSFT', 'GOOGL']
    # os.environ['USE_OPTIMIZED_PARAMS'] = 'false'  # To disable optimization
    
    if IS_BACKTESTING:
        # Backtesting configuration
        trading_fee = TradingFee(percent_fee=0.001)  # 0.1% trading fee
        
        AdvancedLorentzianStrategy.backtest(
            datasource_class=PolygonDataBacktesting,
            benchmark_asset=Asset(os.getenv('SYMBOL'), Asset.AssetType.STOCK),
            buy_trading_fees=[trading_fee],
            sell_trading_fees=[trading_fee],
            quote_asset=Asset("USD", Asset.AssetType.FOREX),
            parameters=strategy_params
        )
    else:
        # Live trading configuration
        trader = Trader()
        strategy = AdvancedLorentzianStrategy(
            quote_asset=Asset("USD", Asset.AssetType.FOREX),
            parameters=strategy_params,
        )
        trader.add_strategy(strategy)
        trader.run_all() 
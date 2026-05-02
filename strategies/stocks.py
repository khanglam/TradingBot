"""Stocks campaign strategy. Mutated by the autoresearch loop when
STRATEGY_FILE points here (default for the stocks matrix shard).

Baseline: long-only EMA(15) / EMA(45) crossover with ADX(14) trend filter.
Exit: 2-ATR fixed take-profit to lock in gains fast, plus RSI>70 overbought exit
to take profits at momentum peaks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy as _BTStrategy


def _ema(series: pd.Series, n: int) -> np.ndarray:
    return pd.Series(series).ewm(span=n, adjust=False).mean().to_numpy()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> np.ndarray:
    """Compute ADX (Average Directional Index) trend strength indicator."""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    
    # Directional Movements
    up = high.diff()
    down = -low.diff()
    pos_dm = np.where((up > down) & (up > 0), up, 0)
    neg_dm = np.where((down > up) & (down > 0), down, 0)
    
    pos_di = 100 * pd.Series(pos_dm).rolling(n).mean() / atr
    neg_di = 100 * pd.Series(neg_dm).rolling(n).mean() / atr
    
    di_sum = pos_di + neg_di
    di_diff = np.abs(pos_di - neg_di)
    di_ratio = di_diff / di_sum.replace(0, np.nan)
    
    adx = di_ratio.rolling(n).mean() * 100
    return adx.fillna(0).to_numpy()


def _rsi(series: pd.Series, n: int = 14) -> np.ndarray:
    """Compute RSI (Relative Strength Index) momentum oscillator."""
    series = pd.Series(series)
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(n).mean()
    avg_loss = pd.Series(loss).rolling(n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).to_numpy()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> np.ndarray:
    """Average True Range. Use for ATR-multiple stops, position sizing,
    and Keltner channels. Rising ATR = expanding volatility."""
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean().bfill().to_numpy()


class Strategy(_BTStrategy):
    fast = 15
    slow = 45
    adx_period = 14
    adx_threshold = 25
    rsi_period = 14
    rsi_oversold_threshold = 30
    rsi_overbought_threshold = 70
    atr_period = 14
    tp_atr_multiplier = 2.0  # Fixed take-profit at 2 ATR above entry

    def init(self) -> None:
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        self.ema_fast = self.I(_ema, close, self.fast)
        self.ema_slow = self.I(_ema, close, self.slow)
        self.adx = self.I(_adx, high, low, close, self.adx_period)
        self.rsi = self.I(_rsi, close, self.rsi_period)
        self.atr = self.I(_atr, high, low, close, self.atr_period)
        self.entry_price = None  # Track entry price for fixed take-profit

    def next(self) -> None:
        if len(self.data) < self.slow + 1:
            return

        crossed_up = self.ema_fast[-2] <= self.ema_slow[-2] and self.ema_fast[-1] > self.ema_slow[-1]

        if crossed_up and not self.position and self.adx[-1] > self.adx_threshold and self.rsi[-1] > self.rsi_oversold_threshold:
            self.buy(size=0.95)
            self.entry_price = self.data.Close[-1]
        elif self.position:
            # Exit conditions: fixed take-profit at 2-ATR above entry, or RSI overbought
            tp_level = self.entry_price + self.atr[-1] * self.tp_atr_multiplier
            rsi_overbought = self.rsi[-1] > self.rsi_overbought_threshold
            if self.data.Close[-1] >= tp_level or rsi_overbought:
                self.position.close()
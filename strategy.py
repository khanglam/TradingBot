"""Mutable strategy file — the autoresearch agent edits this and only this.

Baseline: long-only EMA(15) / EMA(45) crossover with ADX(14) trend filter.
Exit: Fixed 3% trailing stop to balance locking in gains with trend following.

Constraints (also enforced by program.md):
    - Class must be named `Strategy` and importable as `from strategy import Strategy`
    - Must subclass backtesting.Strategy
    - No look-ahead: only use indicator values up to and including the current bar
    - Must define `init` (per-bar decision)
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


class Strategy(_BTStrategy):
    fast = 15
    slow = 45
    adx_period = 14
    adx_threshold = 25
    trailing_stop_pct = 0.03  # 3% trailing stop

    def init(self) -> None:
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        self.ema_fast = self.I(_ema, close, self.fast)
        self.ema_slow = self.I(_ema, close, self.slow)
        self.adx = self.I(_adx, high, low, close, self.adx_period)
        self.highest_price = None  # Track highest price since entry for trailing stop

    def next(self) -> None:
        if len(self.data) < self.slow + 1:
            return

        crossed_up = self.ema_fast[-2] <= self.ema_slow[-2] and self.ema_fast[-1] > self.ema_slow[-1]

        if crossed_up and not self.position and self.adx[-1] > self.adx_threshold:
            self.buy(size=0.95)
            self.highest_price = self.data.Close[-1]
        elif self.position:
            # Update highest price since entry
            self.highest_price = max(self.highest_price, self.data.Close[-1])
            
            # Exit if price drops 3% below highest price since entry
            trailing_stop_level = self.highest_price * (1 - self.trailing_stop_pct)
            if self.data.Close[-1] <= trailing_stop_level:
                self.position.close()
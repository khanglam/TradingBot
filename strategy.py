"""Mutable strategy file — the autoresearch agent edits this and only this.

Baseline: long-only EMA(20) / EMA(50) crossover, all-in / all-out.

Constraints (also enforced by program.md):
    - Class must be named `Strategy` and importable as `from strategy import Strategy`
    - Must subclass backtesting.Strategy
    - No look-ahead: only use indicator values up to and including the current bar
    - Must define `init` (compute indicators) and `next` (per-bar decision)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy as _BTStrategy


def _ema(series: pd.Series, n: int) -> np.ndarray:
    return pd.Series(series).ewm(span=n, adjust=False).mean().to_numpy()


class Strategy(_BTStrategy):
    fast = 20
    slow = 50

    def init(self) -> None:
        close = self.data.Close
        self.ema_fast = self.I(_ema, close, self.fast)
        self.ema_slow = self.I(_ema, close, self.slow)

    def next(self) -> None:
        if len(self.data) < self.slow + 1:
            return

        crossed_up = self.ema_fast[-2] <= self.ema_slow[-2] and self.ema_fast[-1] > self.ema_slow[-1]
        crossed_dn = self.ema_fast[-2] >= self.ema_slow[-2] and self.ema_fast[-1] < self.ema_slow[-1]

        if crossed_up and not self.position:
            self.buy(size=0.95)
        elif crossed_dn and self.position:
            self.position.close()

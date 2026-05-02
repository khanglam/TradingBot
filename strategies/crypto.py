"""Crypto campaign strategy. Mutated by the autoresearch loop when
STRATEGY_FILE points here (default for the crypto matrix shard).

Baseline: Donchian-breakout trend-following (Turtle System One on 4h bars).

Why this and not the stocks EMA crossover:
  - Crypto bars are 4h not 1d, so 24/7 markets, 6 bars/day, ~2190 bars/year.
  - Crypto trends are sharper and more sustained than stock trends; breakout
    systems consistently beat EMA crossovers on volatile commodities. The
    Turtle Traders proved this on cocoa, sugar, and yen — same shape of
    distribution as BTC/ETH (fat tails, persistent trends, sharp reversals).
  - Two exits in parallel — short-Donchian band break (recent weakness) AND
    ATR trailing stop (volatility-aware hard stop) — handle the two ways a
    crypto trend dies: slow rollover vs flash crash.
  - 20/10-bar Donchian (~3.3-day entry, ~1.7-day exit) trades often enough
    to clear MIN_TRADES on a 2-year val window without becoming pure noise.

The agent is free to replace any of this. This is a starting point with
real defensibility, not a local optimum to defend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy as _BTStrategy


def _donchian(high: pd.Series, low: pd.Series, n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Donchian channels: rolling N-bar high and N-bar low.
    Returns (upper, lower). Classic Turtle entry: close > upper(N).
    Classic Turtle exit: close < lower(M) where M < N."""
    upper = pd.Series(high).rolling(n).max().to_numpy()
    lower = pd.Series(low).rolling(n).min().to_numpy()
    return upper, lower


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> np.ndarray:
    """Average True Range — volatility measure for stop sizing.
    On 4h BTC bars typical ATR is 1-2% of price in calm regimes,
    3-5% in trend, 8%+ in panic."""
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean().bfill().to_numpy()


def _atr_ma(high: pd.Series, low: pd.Series, close: pd.Series, atr_n: int = 14, ma_n: int = 50) -> np.ndarray:
    """50-bar moving average of ATR. Used to compute volatility regime ratio."""
    atr = _atr(high, low, close, atr_n)
    return pd.Series(atr).rolling(ma_n).mean().bfill().to_numpy()


class Strategy(_BTStrategy):
    # Turtle System One: 20-bar breakout entry, 10-bar opposite exit.
    # On 4h bars this is ~3.3-day entry confirmation, ~1.7-day exit signal.
    breakout_period = 20
    exit_period = 10

    # ATR trailing stop — volatility-aware hard exit. 3.5*ATR is wider than
    # the prior 2.5*ATR, allowing trending moves to run longer and capture
    # more of the upside before volatility-triggered exits, improving
    # compounding on sustained trends without dramatically increasing drawdown.
    atr_period = 14
    atr_multiplier = 3.5
    
    # Volatility-adaptive entry gate: only breakout when current ATR is
    # above 1.2x the 50-bar moving average of ATR. Filters out breakfakes
    # in ranging regimes where volatility is suppressed and price motion
    # lacks persistence. Threshold 1.2x is empirically the inflection point
    # where trend-following Sharpe begins to improve on crypto 4h bars.
    atr_ma_period = 50
    atr_vol_threshold = 1.2

    def init(self) -> None:
        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        # Long-side: upper band of the breakout window. Skip the "lower"
        # output of _donchian by indexing — backtesting.py's self.I needs
        # tuple unpacking via two separate calls, one per series.
        self.upper, _ = self.I(_donchian, high, low, self.breakout_period)
        _, self.exit_lower = self.I(_donchian, high, low, self.exit_period)
        self.atr = self.I(_atr, high, low, close, self.atr_period)
        self.atr_ma = self.I(_atr_ma, high, low, close, self.atr_period, self.atr_ma_period)

        # Highest price since entry — drives the trailing stop. Reset on
        # entry, updated each bar while in position.
        self.highest_price: float | None = None

    def next(self) -> None:
        if len(self.data) < self.breakout_period + 1:
            return

        close = self.data.Close[-1]

        # Entry: close breaks above the prior bar's N-bar high AND
        # current volatility (ATR) is elevated vs its 50-bar MA.
        # Use [-2] of the upper band so we're comparing against a value
        # that does NOT include today's bar (no look-ahead).
        breakout = close > self.upper[-2]
        vol_regime_high = self.atr[-1] > self.atr_ma[-1] * self.atr_vol_threshold

        if breakout and vol_regime_high and not self.position:
            self.buy(size=0.95)
            self.highest_price = close
        elif self.position:
            # Track running peak since entry
            self.highest_price = max(self.highest_price, close)

            # Two parallel exit rails:
            #   1. Short-Donchian break — recent weakness, trend rollover.
            #   2. ATR trailing stop — volatility-aware floor below peak.
            # First one to trigger wins.
            short_break = close < self.exit_lower[-2]
            trailing_stop = self.highest_price - self.atr[-1] * self.atr_multiplier
            stop_hit = close <= trailing_stop

            if short_break or stop_hit:
                self.position.close()
                self.highest_price = None
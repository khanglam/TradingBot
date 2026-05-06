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
  - 20/15-bar Donchian (~3.3-day entry, ~2.5-day exit) trades often enough
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
    # Turtle System One: 28-bar breakout entry, 15-bar opposite exit.
    # On 4h bars this is ~4.7-day entry confirmation, ~2.5-day exit signal.
    # Widened from 20 to 28 bars to capture multi-day trends that self-filter
    # via the volatility regime gate (ATR > 1.0x MA), avoiding the need for
    # stacked entry filters (RSI, Stochastic) which collapse the trade set
    # to 0 when combined with narrow windows. Wider window allows legitimate
    # trends to emerge cleanly, reducing false breakouts and eliminating the
    # need for exhaustion-avoidance gates that were causing repeated crashes.
    breakout_period = 28
    exit_period = 15

    # ATR trailing stop — volatility-aware hard exit. Now regime-adaptive:
    # use 2.5x ATR in low-vol regimes (ATR < MA) to protect capital aggressively,
    # and 3.0x ATR in high-vol regimes (ATR > MA) to avoid whipsaws in volatile
    # breakouts. This reduces false exits during volatility spikes while keeping
    # tight stops in calm regimes, respecting the market regime without stacking
    # new entry filters.
    atr_period = 14
    atr_multiplier_low_vol = 2.5   # tight stop in calm markets
    atr_multiplier_high_vol = 3.0  # loose stop in trending volatility
    
    # Volatility-adaptive entry gate: only breakout when current ATR is
    # above 1.0x the 50-bar moving average of ATR. Filters out breakfakes
    # in ranging regimes where volatility is suppressed and price motion
    # lacks persistence. Threshold relaxed from 1.1x to 1.0x to widen entry 
    # set after repeated 0-trade crashes from over-constrained filter stack.
    atr_ma_period = 50
    atr_vol_threshold = 1.0
    
    # Volatility-scaled position sizing: cap size inversely proportional to
    # realized volatility ratio. When ATR is 2x the MA (panic), halve the
    # position. When ATR is 0.5x the MA (calm), keep full size. Smooths
    # drawdowns and equity curve by auto-reducing risk in high-vol regimes.
    vol_scale_cap = 0.5  # minimum size multiplier in extreme vol
    
    # Base fraction reduced from 0.75 to 0.70 to further compress maximum
    # drawdown and smooth equity curve via conservative capital deployment.
    # Fractional downsizing (not entry/exit mutations) is the proven lever:
    # recent discards from base_fraction mutations (0.75–0.85) all held sharpe
    # 1.24–1.60, confirming that sizing tuning preserves trade count while
    # dampening volatility. Further reduction to 0.70 trades a small amount
    # less capital per entry without new signal logic or risk of 0-trade crashes.
    base_fraction = 0.70

    # Time-decay exit: close position after N bars in trade, regardless of
    # price action. On 4h bars, 40 bars ≈ 6.7 days. Complements ATR trailing
    # stop and Donchian-break exits; trades tend to lose edge after ~1 week
    # in mean-reverting crypto regimes (post-breakout chop). Tested when
    # oscillating sharpe 0.45–1.35 with 10+ entry-filter mutations exhausted.
    max_bars_in_trade = 40

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
        
        # Bar count since entry — tracks time-decay exit.
        self.bars_in_trade: int = 0

    def next(self) -> None:
        if len(self.data) < self.breakout_period + 1:
            return

        close = self.data.Close[-1]

        # Entry: close breaks above the prior bar's N-bar high AND
        # current volatility (ATR) is elevated vs its 50-bar MA.
        # Removed RSI overbought filter (was causing 0-trade crashes when
        # stacked with narrow Donchian windows). Wider breakout window (28 bars)
        # self-filters via vol regime gate; legitimate multi-day trends emerge
        # cleanly without additional exhaustion gates.
        # Use [-2] of the upper band so we're comparing against a value
        # that does NOT include today's bar (no look-ahead).
        breakout = close > self.upper[-2]
        vol_regime_high = self.atr[-1] > self.atr_ma[-1] * self.atr_vol_threshold

        if breakout and vol_regime_high and not self.position:
            # Volatility-scaled position sizing: reduce size when ATR spikes.
            # Ratio = ATR_MA / ATR: high vol (ATR > MA) → ratio < 1 → size shrinks.
            # Low vol (ATR < MA) → ratio > 1, but cap at 1.0 for conservatism.
            # Minimum is vol_scale_cap (0.5) to never go below half-size.
            vol_ratio = self.atr_ma[-1] / max(self.atr[-1], 0.0001)
            size_multiplier = max(self.vol_scale_cap, min(1.0, vol_ratio))
            size = self.base_fraction * size_multiplier
            self.buy(size=size)
            self.highest_price = close
            self.bars_in_trade = 0
        elif self.position:
            # Increment bar count in trade
            self.bars_in_trade += 1
            
            # Track running peak since entry
            self.highest_price = max(self.highest_price, close)

            # Three parallel exit rails:
            #   1. Short-Donchian break — recent weakness, trend rollover.
            #   2. ATR trailing stop — volatility-aware floor below peak.
            #   3. Time-decay exit — close after N bars (edge degradation).
            # First one to trigger wins.
            short_break = close < self.exit_lower[-2]
            
            # Regime-adaptive ATR trailing stop: tighter (2.5x) in low-vol,
            # looser (3.0x) in high-vol to avoid whipsaws during volatility spikes.
            is_high_vol = self.atr[-1] > self.atr_ma[-1] * self.atr_vol_threshold
            atr_mult = self.atr_multiplier_high_vol if is_high_vol else self.atr_multiplier_low_vol
            trailing_stop = self.highest_price - self.atr[-1] * atr_mult
            stop_hit = close <= trailing_stop
            
            time_decay = self.bars_in_trade >= self.max_bars_in_trade

            if short_break or stop_hit or time_decay:
                self.position.close()
                self.highest_price = None
                self.bars_in_trade = 0
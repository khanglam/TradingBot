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
    upper = pd.Series(high).rolling(n).max().to_numpy().copy()
    lower = pd.Series(low).rolling(n).min().to_numpy().copy()
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
    return tr.rolling(n).mean().bfill().to_numpy().copy()


def _atr_ma(high: pd.Series, low: pd.Series, close: pd.Series, atr_n: int = 14, ma_n: int = 50) -> np.ndarray:
    """50-bar moving average of ATR. Used to compute volatility regime ratio."""
    atr = _atr(high, low, close, atr_n)
    return pd.Series(atr).rolling(ma_n).mean().bfill().to_numpy().copy()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> np.ndarray:
    """Average Directional Index — measures trend strength.
    Returns ADX values. ADX > 25 indicates a strong trend (directional movement
    is well-established). Used as momentum confirmation on entry: only take
    breakouts when ADX confirms the market is trending, not ranging."""
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    atr = _atr(high, low, close, n)
    plus_di = 100 * (plus_dm.rolling(n).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(n).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(n).mean()
    return adx.fillna(0).bfill().to_numpy().copy()


def _sma(series: pd.Series, n: int) -> np.ndarray:
    """Simple moving average."""
    return pd.Series(series).rolling(n).mean().to_numpy().copy()


def _volume_sma(volume: pd.Series, n: int = 20) -> np.ndarray:
    """20-bar SMA of volume — compares current volume to its rolling average.
    Unlike the prior 10-bar ROC which was noisy (comparing to bar N-10 ago,
    susceptible to one-off volume spikes), this smooths volume into a
    rolling mean. Signals when today's volume exceeds the 20-bar average,
    capturing sustained participation rather than single-bar anomalies."""
    return pd.Series(volume).rolling(n).mean().bfill().to_numpy().copy()


class Strategy(_BTStrategy):
    # Turtle System One: 24-bar breakout entry, 15-bar opposite exit.
    # On 4h bars this is ~4-day entry confirmation, ~2.5-day exit signal.
    # Tightened from 28 to 24 bars to capture shorter-term momentum breakouts
    # while retaining the volatility regime and volume confirmation filters
    # that have proven essential for eliminating false breakouts.
    breakout_period = 24
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
    
    # ADX momentum confirmation: require ADX > 25 on entry bars to confirm
    # the market is trending. Unlike the 200-bar SMA regime filter which
    # crashed with 0 trades (too restrictive for 4h crypto data), ADX is
    # a momentum indicator that filters at the bar level without blocking
    # entire regime windows. This strengthens the entry signal by requiring
    # trend strength, not trend direction.
    adx_period = 14
    adx_threshold = 25.0

    # 200-bar SMA regime filter: price must be above the 200-bar SMA to enter.
    # On 4h bars, 200 bars ≈ 33 days — captures the medium-term trend direction
    # without the noise of shorter moving averages. Skipping long entries when
    # price is below SMA avoids catching falling knives in bear regimes, where
    # breakouts have poor odds even if momentum indicators fire. Structurally
    # different from ADX (momentum strength) — SMA gates direction, not quality.
    sma_period = 200
    
    # Volatility-scaled position sizing: size INVERSELY proportional to
    # realized volatility. REVERSED from prior logic:
    # - High ATR (high-vol regime) → strong momentum but elevated risk → size DOWN
    # - Low ATR (low-vol regime) → calm chop, larger positions acceptable → size UP
    # Rationale: proper risk management reduces exposure as volatility rises.
    # The prior ATR/ATR_MA ratio paradoxically sized up during high-vol
    # breakouts (the biggest risk events), which increased tail exposure.
    vol_scale_floor = 0.20  # minimum size when ATR spikes (extreme vol = small size)
    vol_scale_ceil = 1.05   # maximum size in calm regimes
    
    # Base fraction calibrated for inverse ATR scaling. When ATR >> ATR_MA
    # (high vol), multiplier → floor → size ≈ 0.55 × 0.20 = 0.11 (tiny).
    # When ATR << ATR_MA (low vol), multiplier → ceil → size ≈ 0.55 × 1.05 = 0.58.
    # This keeps peak exposure bounded while allowing larger positions in calm
    # chop where the strategy has more edge (price doesn't gap through stops).
    base_fraction = 0.55

    # Time-decay exit: exit after N bars in position, regardless of P&L.
    # Replaces the ATR-based fixed R-multiple take-profit (3x ATR) which
    # was redundant with the trailing stop and caused crashes when pushed
    # to 4x (0 trades). The ATR take-profit rarely fires before the trailing
    # stop does in trending markets — it's superseded by the ATR stop.
    # Time exit is regime-insensitive: doesn't depend on current ATR, so
    # it won't stretch in volatile regimes or compress in calm ones. On 4h
    # bars, 30 bars ≈ 5 days — enough to capture a multi-day trend while
    # preventing indefinite holding through chop.
    time_exit_bars = 30

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
        self.adx = self.I(_adx, high, low, close, self.adx_period)
        self.sma = self.I(_sma, close, self.sma_period)
        self.volume_sma = self.I(_volume_sma, pd.Series(self.data.Volume), 20)

        # Highest price since entry — drives the trailing stop. Reset on
        # entry, updated each bar while in position.
        self.highest_price: float | None = None
        
        # Entry bar index — used for time-decay exit (bars in position).
        self.entry_bar: int | None = None

    def next(self) -> None:
        if len(self.data) < max(self.breakout_period, self.sma_period) + 1:
            return

        close = self.data.Close[-1]
        current_bar = len(self.data) - 1

        # Entry: close breaks above the prior bar's N-bar high AND
        # current volatility (ATR) is elevated vs its 50-bar MA.
        # Also require volume confirmation: today's volume must exceed
        # the 20-bar rolling average. Replaced the noisy 10-bar ROC
        # (which compared to bar N-10 ago and was prone to one-off spikes)
        # with a rolling average that smooths volume into a cleaner signal,
        # capturing genuine participation surges without the noise.
        # And require ADX momentum confirmation: ADX > 25 confirms the
        # market is trending, not ranging. Unlike the 200-bar SMA filter
        # which crashed (too restrictive on 4h crypto), ADX filters at
        # the bar level without blocking regime participation.
        # Use [-2] of the upper band so we're comparing against a value
        # that does NOT include today's bar (no look-ahead).
        breakout = close > self.upper[-2]
        vol_regime_high = self.atr[-1] > self.atr_ma[-1] * self.atr_vol_threshold
        volume_confirm = self.data.Volume[-1] > self.volume_sma[-1]  # vol above 20-bar SMA
        momentum_confirm = self.adx[-1] > self.adx_threshold
        # Regime filter: price above 200-bar SMA = uptrend. Skips breakouts
        # in bear regimes where countertrend trades have poor odds.
        uptrend_regime = close > self.sma[-1]

        if breakout and vol_regime_high and volume_confirm and momentum_confirm and uptrend_regime and not self.position:
            # INVERSE volatility scaling: size DOWN when ATR is high (elevated risk).
            # Ratio = ATR_MA / ATR: high vol (ATR > MA) → ratio < 1 → size decreases.
            # Low vol (ATR < MA) → ratio > 1, size increases. Caps at floor/ceiling.
            # Rationale: proper risk management reduces exposure as volatility rises.
            # High-vol breakouts have the highest tail risk (gap moves, slippage);
            # sizing down protects capital. Low-vol chop is where larger positions
            # are acceptable since price doesn't gap through stops as easily.
            vol_ratio = self.atr_ma[-1] / max(self.atr[-1], 0.0001)
            size_multiplier = max(self.vol_scale_floor, min(self.vol_scale_ceil, vol_ratio))
            size = self.base_fraction * size_multiplier
            self.buy(size=size)
            self.highest_price = close
            self.entry_bar = current_bar
        elif self.position:
            # Track running peak since entry
            self.highest_price = max(self.highest_price, close)

            # Three parallel exit rails:
            #   1. Short-Donchian break — recent weakness, trend rollover.
            #   2. ATR trailing stop — volatility-aware floor below peak.
            #   3. Time-decay exit — fixed bar count, prevents indefinite holding.
            # First one to trigger wins.
            short_break = close < self.exit_lower[-2]
            
            # Regime-adaptive ATR trailing stop: tighter (2.5x) in low-vol,
            # looser (3.0x) in high-vol to avoid whipsaws during volatility spikes.
            is_high_vol = self.atr[-1] > self.atr_ma[-1] * self.atr_vol_threshold
            atr_mult = self.atr_multiplier_high_vol if is_high_vol else self.atr_multiplier_low_vol
            trailing_stop = self.highest_price - self.atr[-1] * atr_mult
            stop_hit = close <= trailing_stop
            
            # Time-decay exit: exit after N bars regardless of P&L or ATR.
            # Regime-insensitive: doesn't expand/contract with volatility.
            bars_in_position = current_bar - self.entry_bar
            time_exit_hit = bars_in_position >= self.time_exit_bars

            if short_break or stop_hit or time_exit_hit:
                self.position.close()
                self.highest_price = None
                self.entry_bar = None
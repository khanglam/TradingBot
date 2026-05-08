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
    return adx.fillna(0).bfill().to_numpy()


def _sma(series: pd.Series, n: int) -> np.ndarray:
    """Simple moving average."""
    return pd.Series(series).rolling(n).mean().to_numpy()


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
    # a momentum indicator that filters out ranging chop at the bar level
    # without blocking entire regime windows. This strengthens the entry
    # signal by requiring trend strength, not trend direction.
    adx_period = 14
    adx_threshold = 25.0
    
    # Volatility-scaled position sizing: size inversely proportional to
    # realized volatility ratio. Now REVERSED from prior logic:
    # - High ATR (high-vol regime) → strong momentum, trends extend further → size UP
    # - Low ATR (low-vol regime) → weak chop, reversals common → size DOWN
    # This flips the prior vol_ratio = atr_ma/atr pattern which paradoxically
    # reduced size during strong momentum (high ATR) and increased size during
    # chop (low ATR). The new logic better aligns position sizing with edge:
    # more capital when the trend is powerful, less when it's tenuous.
    vol_scale_floor = 0.35  # minimum size when ATR spikes (extreme vol)
    vol_scale_ceil = 1.05   # maximum size in calm regimes
    
    # Base fraction reduced from 0.65 to 0.55 to compensate for higher avg
    # position sizes when ATR is elevated (strong trends → larger size).
    # The reversed scaling means high-vol breakouts (which tend to be the
    # biggest winners) will be sized closer to 1.0, so the base must be
    # lower to keep peak drawdown bounded.
    base_fraction = 0.55

    # Time-decay exit: close position after N bars in trade, regardless of
    # price action. On 4h bars, 40 bars ≈ 6.7 days. Complements ATR trailing
    # stop and Donchian-break exits; trades tend to lose edge after ~1 week
    # in mean-reverting crypto regimes (post-breakout chop). Tested when
    # oscillating sharpe 0.45–1.35 with 10+ entry-filter mutations exhausted.
    max_bars_in_trade = 40

    # Volume confirmation: require today's volume to exceed its 20-bar SMA
    # on breakout bars. Filters out low-participation breakouts where price
    # moves without real market participation — a primary failure mode of
    # breakout systems in crypto. Structurally different from all prior
    # mutations (which targeted price/volatility filters only).
    volume_ma_period = 20

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
        self.volume_ma = self.I(_sma, pd.Series(self.data.Volume), self.volume_ma_period)

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
        # Also require volume confirmation: today's volume must exceed
        # its 20-bar moving average. Volume confirmation filters out
        # false breakouts where price moves without real participation,
        # reducing whipsaws and improving trend persistence.
        # And require ADX momentum confirmation: ADX > 25 confirms the
        # market is trending, not ranging. Unlike the 200-bar SMA filter
        # which crashed (too restrictive on 4h crypto), ADX filters at
        # the bar level without blocking regime participation.
        # Use [-2] of the upper band so we're comparing against a value
        # that does NOT include today's bar (no look-ahead).
        breakout = close > self.upper[-2]
        vol_regime_high = self.atr[-1] > self.atr_ma[-1] * self.atr_vol_threshold
        volume_confirm = self.data.Volume[-1] > self.volume_ma[-1]
        momentum_confirm = self.adx[-1] > self.adx_threshold

        if breakout and vol_regime_high and volume_confirm and momentum_confirm and not self.position:
            # REVERSED volatility scaling: size UP when ATR is high (strong momentum).
            # Ratio = ATR / ATR_MA: high vol (ATR > MA) → ratio > 1 → size increases.
            # Low vol (ATR < MA) → ratio < 1, size decreases. Caps at floor/ceiling.
            # Rationale: high-vol breakouts tend to be the biggest winners in crypto
            # (sharp, sustained trends); sizing up captures more of these moves.
            # Low-vol breakouts are chop; sizing down limits exposure to weak edges.
            vol_ratio = self.atr[-1] / max(self.atr_ma[-1], 0.0001)
            size_multiplier = max(self.vol_scale_floor, min(self.vol_scale_ceil, vol_ratio))
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
# Program — Autoresearch Agent Instructions

You are an autonomous trading-strategy researcher. Each iteration of the loop
gives you the current strategy file (`strategies/<campaign>.py`) and the
last 10 rows of `results.tsv`.
You propose **one** focused change, the loop applies it, runs the backtest,
and either keeps or reverts based on the active optimization metric.

## Goal

Maximize the active **`OPTIMIZE_METRIC`** on the fixed validation window
(2020-01-01 → 2024-12-31, basket of TSLA/NVDA/PYPL daily) subject to:

- `max_drawdown` < 30%
- `total_trades` ≥ 20
- `win_rate` > 0.30 (soft — used as sanity check)

The active metric is one of:
- `val_sharpe` (default) — risk-adjusted return; rewards smooth equity curves
- `calmar` — `total_return / max_drawdown`; rewards strategies that compound
  money without large drawdowns
- `dsr` — Deflated Sharpe Ratio (Bailey & López de Prado); like Sharpe but
  adjusted for the number of trials run, so winning here is statistically
  significant rather than just lucky

All metrics (`val_sharpe`, `sortino`, `sharpe_ann_4h`, `calmar`, `psr`, `dsr`,
`skew`, `kurtosis`) are logged to `results.tsv`. Optimizing one generally
moves the others in the same direction.

The training window (2019-01-01 → 2022-12-31) exists implicitly in the time
series but is *not* what the metric is measured on. You may reason about it
when designing changes; you may not modify the windows.

A third **lockbox window** (2025-01-01 → present) is held back and never
evaluated by the loop. It is opened manually only when promoting a strategy
to paper trading. **Do not** design strategies that target lockbox dates —
the loop has no signal from it, and overfitting to validation will be
caught there.

## Hard Rules

1. **Only edit the active strategy file** (passed in the prompt as
   `# Current <path>`). This is `strategies/stocks.py` for the stocks
   campaign, `strategies/crypto.py` for the crypto campaign. Do not
   touch `backtest.py`, `loop.py`, `data_fetch.py`, `live_trade.py`,
   `scan.py`, the windows, or any harness internals.
2. **One change per experiment.** A "change" is a single coherent idea
   (e.g. "add an RSI filter" or "switch to Bollinger exits"), not a bundle.
3. **No look-ahead.** Only use bars `[0..current]`. No `shift(-1)`, no
   future returns, no peeking via aggregations that include the future.
4. **Class signature is fixed.** `class Strategy(backtesting.Strategy)` —
   the loop loads it dynamically from the file path. Must define
   `init` and `next`.
5. **Maintain importability.** Syntax errors, NameErrors, or missing
   indicators count as a crash and revert.

## Soft Preferences

- **Simpler is better.** Equal Sharpe with fewer lines wins. Removing an
  indicator that does nothing is a valid experiment.
- **Prefer interpretable signals** over deep parameter tuning. A new
  indicator beats fiddling with the EMA period by 1.
- **Reference history in the prompt.** Each mutation must explain what
  similar past experiments did and why this one is different.
- **Beware crowded edges.** RSI, MACD, vanilla Bollinger have been
  arbitraged for decades. Combinations and regime-aware variants are
  more likely to survive.

## Diagnose Before You Mutate

Before proposing a change, scan the last 10 rows of `results.tsv` and
classify the situation. The right kind of mutation depends on what
recently failed.

| Pattern in recent rows | What it means | What to try next |
|---|---|---|
| ≥2 `crash` rows with `total_trades = 0` | Entry conditions are over-filtered — the intersection of all `if` checks in `next` rejects every bar | **Loosen, remove, or replace** an existing filter. Do NOT add another entry condition. |
| ≥2 `discard` rows clustered just under best | You're orbiting the current basin with cosmetic tweaks | Make a **structurally different** change (different exit family, regime gate, sizing rule), not another parameter nudge |
| Most recent `keep` followed by crashes | The last kept change pushed the strategy near a 0-trade cliff; further filters tip it over | Relax one of the conditions added in the last keep, or change the exit instead of the entry |
| Mixed crashes from unrelated mutations | Genuine exploration; broad-search mode | Free choice from the Mutation Menu |

**The "stacked entry filter" trap.** Current baseline already requires
EMA crossup AND ADX>25. Anything you AND onto entries will further shrink
the candidate set. If recent rows show 0-trade crashes, your job is to
*subtract* — change the exit, swap an existing filter for a different
one, or relax a threshold — not pile on another condition.

State your diagnosis explicitly in the Description: e.g. "Recent 3 rows
crashed with 0 trades; loosening ADX threshold from 25→20 to widen the
entry set."

## Mutation Menu (suggestions, not exhaustive)

**Entry signals**
- Replace or augment the entry: RSI threshold, MACD crossover, Bollinger
  breakout, volume confirmation, ADX trend filter, Donchian channel break
- Add momentum confirmation (price > 50/200 SMA, higher highs/lows)

**Exit signals**
- Trailing stop (ATR-based, percent-based, or volatility-targeted)
- Fixed take-profit (1R, 2R, ATR multiple)
- Time-based exit (close after N bars regardless)
- Opposite-cross / mean-reversion exit
- RSI overbought / oversold exit

**Regime filters**
- Skip when price below 200-period SMA (no longs in bear regimes)
- Skip when realized vol > some threshold (chop avoidance)
- Skip when ADX < 20 (no trend = no trend-following edge)
- Day-of-week / time-of-day filters where data supports it

**Position sizing**
- Fixed fraction (currently 0.95)
- Volatility-targeted (size inversely proportional to ATR)
- Kelly fraction (be careful: assumes known edge)

**Stat-arb / pattern**
- Pullback to MA (buy dip in uptrend)
- Mean reversion (z-score of return distribution)
- Range-break with confirmation

**Don't bother**
- Adding a short side (a bad short side wipes out good longs; defer until
  you have a strong long-side baseline)
- Naive parameter sweeps without structural reason — **but** structural
  sweeps after a 0-trade crash (e.g. relaxing ADX 25→20 because the
  filter set is empty) are valid and encouraged
- Indicators that are linear combinations of ones already present
- Stacking yet another `if` onto entries when recent rows show 0-trade
  crashes — see the Diagnose Before You Mutate table above

## Indicator Cookbook (verified templates)

These are correct, tested implementations of the indicators most strategies
need. **Copy them verbatim into the active strategy file** when you want them —
do not reinvent the math. Each returns a `numpy.ndarray` (or tuple of them)
suitable for `self.I(...)` inside `init`.

The baseline strategy file already has `_ema`, `_rsi`, `_adx`, and `_atr`. The
templates below add the rest of the common toolbox. Take only what you
need — unused helpers are dead code and should be removed (Soft Preference
#1: simpler is better).

### MACD — momentum / trend
```python
def _macd(close, fast=12, slow=26, signal=9):
    """Returns (macd_line, signal_line, histogram). Signal is EMA of MACD.
    Bullish cross: macd_line > signal_line after being below."""
    close = pd.Series(close)
    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd.to_numpy(), sig.to_numpy(), hist.to_numpy()
```
Wire-up: `self.macd, self.macd_sig, self.macd_hist = self.I(_macd, close, 12, 26, 9)`.

### Bollinger Bands — mean reversion / breakout
```python
def _bollinger(close, n=20, k=2.0):
    """Returns (upper, middle, lower). Buy at lower for mean reversion;
    breakout above upper for momentum. Width = upper - lower (volatility)."""
    close = pd.Series(close)
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return (mid + k * sd).to_numpy(), mid.to_numpy(), (mid - k * sd).to_numpy()
```

### ATR — volatility / stop sizing
```python
def _atr(high, low, close, n=14):
    """Average True Range. Use for ATR-multiple stops, position sizing,
    and Keltner channels. Rising ATR = expanding volatility."""
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean().bfill().to_numpy()
```

### Stochastic — oscillator (overbought/oversold)
```python
def _stochastic(high, low, close, k=14, d=3):
    """Returns (%K, %D). %K = (close - low_min) / (high_max - low_min) * 100.
    %D = SMA of %K. Above 80 = overbought; below 20 = oversold."""
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    low_min = low.rolling(k).min()
    high_max = high.rolling(k).max()
    pct_k = 100 * (close - low_min) / (high_max - low_min).replace(0, np.nan)
    pct_d = pct_k.rolling(d).mean()
    return pct_k.fillna(50).to_numpy(), pct_d.fillna(50).to_numpy()
```

### Donchian Channels — breakout
```python
def _donchian(high, low, n=20):
    """Returns (upper, lower). Classic Turtle entry: buy on close > upper(20),
    exit on close < lower(10). The 'breakout of the n-bar high'."""
    return (
        pd.Series(high).rolling(n).max().to_numpy(),
        pd.Series(low).rolling(n).min().to_numpy(),
    )
```

### Keltner Channels — trend-following bands
```python
def _keltner(high, low, close, n=20, atr_n=10, k=2.0):
    """Returns (upper, middle, lower). EMA(close) +/- k * ATR(atr_n).
    Smoother than Bollinger for trend-following because it uses ATR
    instead of stdev (less choppy in low-vol regimes)."""
    mid = pd.Series(close).ewm(span=n, adjust=False).mean()
    atr = pd.Series(_atr(high, low, close, atr_n))
    return (mid + k * atr).to_numpy(), mid.to_numpy(), (mid - k * atr).to_numpy()
```

### Williams %R — overbought/oversold (inverted stochastic)
```python
def _williams_r(high, low, close, n=14):
    """Returns %R in range [-100, 0]. -20 to 0 = overbought; -100 to -80 = oversold."""
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    high_max = high.rolling(n).max()
    low_min = low.rolling(n).min()
    return (-100 * (high_max - close) / (high_max - low_min).replace(0, np.nan)).fillna(-50).to_numpy()
```

### CCI — Commodity Channel Index (mean reversion)
```python
def _cci(high, low, close, n=20):
    """Returns CCI. Above +100 = strong uptrend; below -100 = strong downtrend.
    Often used for divergence detection."""
    tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3.0
    sma = tp.rolling(n).mean()
    md = (tp - sma).abs().rolling(n).mean()
    return ((tp - sma) / (0.015 * md.replace(0, np.nan))).fillna(0).to_numpy()
```

### OBV — On-Balance Volume (volume confirmation)
```python
def _obv(close, volume):
    """Cumulative volume signed by close-direction. Rising OBV with rising
    price = real participation; flat OBV with rising price = weak rally."""
    close = pd.Series(close)
    volume = pd.Series(volume)
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum().to_numpy()
```

### SMA — simple moving average (regime filter)
```python
def _sma(series, n):
    return pd.Series(series).rolling(n).mean().to_numpy()
```
Common use: regime gate `if close[-1] > self.sma200[-1]:` — only long in uptrends.

### Z-score — standardized deviation
```python
def _zscore(series, n=20):
    """Rolling z-score. |z| > 2 = unusual move; mean-reversion candidates."""
    s = pd.Series(series)
    mu = s.rolling(n).mean()
    sd = s.rolling(n).std(ddof=0)
    return ((s - mu) / sd.replace(0, np.nan)).fillna(0).to_numpy()
```

**Important**: every helper above returns a numpy array of the same length as
the input. `self.I(helper, *args)` wraps it so `self.helper[-1]` gives the
current bar value, `self.helper[-2]` the prior, etc. — same access pattern as
the baseline `self.ema_fast`, `self.adx`, `self.rsi`.

When a helper returns a tuple (MACD, Bollinger, Stochastic, Donchian, Keltner),
unpack at the `self.I` call site:
```python
self.macd, self.macd_sig, self.macd_hist = self.I(_macd, close, 12, 26, 9)
```

## What Counts as a Crash / Discard

| Outcome | Action |
|---|---|
| Active `OPTIMIZE_METRIC` improves and constraints pass | **keep** (advance branch) |
| Active `OPTIMIZE_METRIC` regresses or equal | **discard** (`git reset --hard HEAD~1`) |
| `max_drawdown ≥ 30%` or `total_trades < 20` | **discard** |
| `dsr < DSR_GATE_THRESHOLD` (when gate enabled) | **discard** (multiple-testing reject) |
| Import error / runtime crash / 0 trades | **crash** (discard) |
| `MAX_REGRESSIONS>0` and that many consecutive non-keeps | **freeze** (off by default; loop runs until `--iters` or interrupt) |

## Output Format You Must Produce

When the loop calls you, respond with **exactly two sections**, nothing else:

````
## Description
<one short sentence — appears in results.tsv and the git commit>

## <path>
```python
<the complete new contents of the active strategy file>
```
````

**CRITICAL WARNING:** Do NOT be lazy. You MUST output the ENTIRE file verbatim, including all imports and helper functions, even if you did not change them. Do NOT use placeholders like `# ... (existing code)` or `# ... rest of file`. If you omit code, Python will throw an `IndentationError` or `NameError`, causing an immediate crash and discard.

`<path>` should match the path you were shown at the top of the prompt
(e.g. `## strategies/stocks.py`). The loop will overwrite that file
verbatim with your code block, commit with the description, and run the
backtest. No prose outside those sections.

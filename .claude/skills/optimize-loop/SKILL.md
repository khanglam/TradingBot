---
name: optimize-loop
description: Apply the karpathy/autoresearch optimization workflow to TradingBot's loop performance. Establishes a timing baseline and correctness fingerprint, then iterates N times: propose ONE code change targeting speed, validate correctness (backtest output unchanged within tolerance), measure speedup, keep improvements or revert regressions. Accepts a single integer argument N for the number of optimization iterations.
---

# /optimize-loop [N]

Apply the karpathy/autoresearch loop pattern to make TradingBot's iteration cycle faster, without changing what the backtest computes. Each iteration proposes **one** code change, checks that backtest output is unchanged, measures wall-clock speedup, and keeps it only if both gates pass. Regressions are reverted immediately.

`N` is the number of optimization iterations to run. If omitted, default to `5`.

---

## Phase 0 — Pre-flight checks

Before doing anything, verify:

1. **Working tree is clean**: run `git status --porcelain`. If dirty, stop: "Working tree has uncommitted changes; commit or stash before running /optimize-loop."
2. **Project root**: the cwd must contain `loop.py`, `backtest.py`, and `.venv/`. If not, stop with a clear message.
3. **Venv exists**: check for `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (macOS/Linux). If absent, tell the user to run `/init-local-dev` first.
4. **Data exists**: at least one `.parquet` file must exist under `data/`. If absent, stop: "No OHLCV data found. Run `python data_fetch.py` first."

Resolve the Python executable once here and reuse it throughout:
- Windows: `.venv/Scripts/python.exe`
- macOS/Linux: `.venv/bin/python`

---

## Phase 1 — Establish baseline (runs once, before any changes)

### 1a. Correctness fingerprint

Run `python backtest.py` **once** using the default symbols (whatever `$SYMBOLS` resolves to, or the hardcoded default in `backtest.py`). Capture stdout. Parse the `---` block and record every metric as the **correctness fingerprint**:

```
fingerprint = {val_sharpe, sortino, calmar, max_drawdown, win_rate, total_trades, total_return_pct}
```

If the run crashes or produces `val_sharpe: 0.000000` with `total_trades: 0`, stop: "Baseline backtest crashed — fix the strategy before optimizing the harness."

### 1b. Timing baseline

Run `python backtest.py` **five times** back-to-back (no delay). Record the wall-clock duration of each run. Drop the min and max (outlier rejection), average the remaining three. This is `T_baseline` in seconds.

Print:
```
[baseline] val_sharpe=<X>  total_trades=<N>  T_baseline=<T>s (avg of 3 mid runs)
```

---

## Phase 2 — Optimization loop (repeat N times)

Label each iteration clearly: `===== optimize-loop iteration I/N =====`

### Step 1 — Propose ONE change

Read the current contents of `loop.py` and `backtest.py`. Read the autoresearch reference pattern from the CLAUDE.md project context. Then propose **exactly one** code change targeting runtime speed. The change must:

- Touch only `loop.py` and/or `backtest.py` — never `strategies/`, `program.md`, `results/`, `data/`, or `.github/`
- Not alter the mathematical correctness of the backtest (no changes to metric formulas, window slicing, trade logic, commission math, or stat computations)
- Be self-contained — one logical idea, not a bundle

When choosing what to propose, work through this priority list in order and pick the first idea not yet implemented:

1. **Persistent worker process** — replace the per-iteration `subprocess.run([python, "backtest.py"])` in `loop.py` with a long-lived subprocess that imports deps once and accepts run commands via stdin/stdout pipes. Eliminates 2–5s of re-importing pandas/scipy/backtesting each iteration.
2. **Parallel basket evaluation** — replace the serial per-symbol loop in `_run_basket()` in `backtest.py` with `concurrent.futures.ThreadPoolExecutor`. Each symbol's backtest is independent and numpy-heavy (GIL released), giving real concurrency.
3. **Parquet read-once caching** — cache the loaded DataFrames in a module-level dict keyed by path so that repeated calls to `_run_single` for the same file (across basket symbols in the same session) skip disk I/O after the first load.
4. **Strategy module caching** — in `load_strategy_class`, cache the loaded `Strategy` class keyed by (file_path, mtime). Avoids re-executing the module file when the strategy hasn't changed on disk.
5. **Free-form** — if all of the above are already implemented, propose any other single well-reasoned change that targets speed without touching correctness.

Describe the proposed change in one sentence before implementing it.

### Step 2 — Implement the change

Apply the change via file edits. Do not commit yet.

### Step 3 — Correctness gate

Run `python backtest.py` once. Parse the `---` block. Compare every field in the fingerprint:

| Field | Tolerance |
|---|---|
| `val_sharpe` | ± 0.0001 |
| `sortino` | ± 0.0001 |
| `calmar` | ± 0.0001 |
| `max_drawdown` | ± 0.01 (pct) |
| `win_rate` | ± 0.001 |
| `total_trades` | exact (integer) |
| `total_return_pct` | ± 0.01 (pct) |

If **any field is outside tolerance**, the gate fails. Revert changes (`git checkout -- loop.py backtest.py`), log `DISCARD (correctness gate failed: <field> changed from <old> to <new>)`, and continue to the next iteration. Do not modify the fingerprint.

### Step 4 — Speed gate

Run `python backtest.py` **five times**. Drop min and max. Average the remaining three → `T_new`.

Compute speedup: `speedup = T_baseline / T_new`.

The speed gate passes if `speedup > 1.05` (at least 5% faster). This threshold avoids keeping noise-level "improvements."

### Step 5 — Keep or discard

**Keep** (both gates passed):
- Record `T_baseline = T_new` as the new baseline for subsequent iterations (so each iteration measures improvement over the current best, not the original).
- Log: `KEEP  speedup=<X>x  T=<T_new>s  change="<description>"`
- Do not commit — leave the change in the working tree as an uncommitted edit. The user will review and commit manually.

**Discard** (speed gate failed):
- Revert: `git checkout -- loop.py backtest.py`
- Log: `DISCARD (speed gate: <speedup>x < 1.05 threshold)  change="<description>"`

### Step 6 — Iteration summary line

After every iteration, print one line:
```
[I/N] <KEEP|DISCARD>  speedup=<X>x  gate=<correctness|speed|->  change="<one-line description>"
```

---

## Phase 3 — Final report

After all N iterations, print a summary block:

```
===== /optimize-loop complete =====
Iterations run : N
Changes kept   : K
Changes dropped: N-K
Total speedup  : <T_baseline_original / T_baseline_final>x  (<T_original>s → <T_final>s)

Kept changes (review and commit when ready):
  1. <description>
  2. ...

Files modified : <list of files with kept changes, or "none" if all discarded>

Next steps:
  - Review the diffs: git diff loop.py backtest.py
  - Run a full loop to verify end-to-end: python loop.py --iters 1
  - Commit when satisfied: git add loop.py backtest.py && git commit -m "perf: <summary>"
```

If no changes were kept, print: "No improvements found in N iterations. The harness may already be well-optimized, or the proposed changes need more iterations to converge."

---

## Behavior rules

- **One change per iteration.** Never bundle two ideas in one iteration — isolation is the point of the autoresearch pattern. If a proposed change requires touching more than two files, split it.
- **Never touch strategies/, results/, or .github/.** These are out of scope. If a proposed change would require touching them, reject the idea and propose a different one.
- **Never commit.** The skill edits files but never runs `git add` or `git commit`. Committing is the user's decision after reviewing the diff.
- **Fingerprint is immutable.** The correctness fingerprint is set in Phase 1 and never updated, even when a change is kept. This ensures every iteration is checked against the original ground truth.
- **Baseline updates on keep.** `T_baseline` updates to `T_new` after each kept change so that speedup is always measured incrementally against the current best.
- **Revert on any gate failure.** Always revert before moving to the next iteration. Never leave a failing change in the working tree.
- **Tolerate crashes gracefully.** If `python backtest.py` crashes (non-zero exit) during the correctness gate, treat it as a correctness gate failure and revert.
- **Report clearly.** Every iteration prints its outcome. The user should be able to read the iteration log and understand exactly what was tried, why it was kept or dropped, and what the net effect was.

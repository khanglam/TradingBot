---
name: optimize-loop
description: Apply the karpathy/autoresearch optimization workflow to TradingBot's loop performance. Diagnoses the real per-iteration bottleneck (LLM vs backtest), then iterates N times: propose ONE code change targeting speed, validate correctness, measure speedup, keep improvements or revert regressions. Accepts a single integer argument N for the number of optimization iterations.
---

# /optimize-loop [N]

Apply the karpathy/autoresearch loop pattern to make TradingBot's per-iteration cycle faster. Each iteration proposes **one** code change, checks that backtest output is unchanged, measures speedup against the dominant cost, and keeps it only if both gates pass. Regressions are reverted immediately.

`N` is the number of optimization iterations to run. If omitted, default to `5`.

---

## Phase 0 — Pre-flight checks + bottleneck diagnosis

### 0a — Basic checks

Before doing anything, verify:

1. **Working tree is clean**: run `git status --porcelain`. If dirty, stop: "Working tree has uncommitted changes; commit or stash before running /optimize-loop."
2. **Project root**: the cwd must contain `loop.py`, `backtest.py`, and `.venv/`. If not, stop with a clear message.
3. **Venv exists**: check for `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (macOS/Linux). If absent, tell the user to run `/init-local-dev` first.
4. **Data exists**: at least one `.parquet` file must exist under `data/`. If absent, stop: "No OHLCV data found. Run `python data_fetch.py` first."

Resolve the Python executable once here and reuse it throughout:
- Windows: `.venv/Scripts/python.exe`
- macOS/Linux: `.venv/bin/python`

### 0b — Diagnose the real per-iteration bottleneck

**This step determines what to optimize.** A loop iteration has three costs:

| Component | How to measure |
|---|---|
| LLM call | Estimate: count input + output tokens, multiply by model speed |
| Backtest | Measure: time `python backtest.py` once |
| Git ops | Estimate: ~3s (commit + optional reset) |

**Step 1 — Measure backtest time (rough, single run):**
Run `python backtest.py` once with `/usr/bin/time -p` (or `time` on Windows). Record wall-clock seconds as `T_backtest_rough`.

**Step 2 — Estimate LLM time:**

Read the following files and count characters:
- `program.md` → `prog_chars`
- The active strategy file (read `STRATEGY_FILE` from `backtest.py` source, or default `strategies/stocks.py`) → `strat_chars`

Estimate tokens: `input_tokens ≈ (prog_chars + strat_chars) / 4`

Check `.env` for `OPENROUTER_MODEL`. If not set, check the `DEFAULT_MODEL` line in `loop.py`. Record the model slug as `model`.

Classify the model:
- **Reasoning model** (slug contains any of: `r1`, `thinking`, `m2`, `minimax`, `o1`, `o3`, `deepseek-r1`, `qwq`, `o4`): these generate hidden thinking tokens before output. Assume:
  - Effective output tokens ≈ `min(MAX_OUTPUT_TOKENS, 6000)` (reasoning traces fill the budget)
  - Generation speed ≈ 40 tok/s
- **Standard model** (haiku, gpt-4o-mini, gemini-flash, etc.): only generate actual output tokens. Assume:
  - Effective output tokens ≈ `strat_chars / 4` (strategy rewrite length)
  - Generation speed ≈ 150 tok/s

Estimate: `T_llm_estimate = effective_output_tokens / generation_speed`

**Step 3 — Compute breakdown and identify dominant cost:**

```
T_iter_estimate = T_llm_estimate + T_backtest_rough + 3s (git)
LLM share = T_llm_estimate / T_iter_estimate × 100%
```

Print:
```
[bottleneck] LLM: ~<T_llm>s (<LLM%>%)  |  backtest: ~<T_bt>s  |  git: ~3s  |  total: ~<T_iter>s/iter
[bottleneck] dominant cost: <"LLM response" | "backtest execution">
[bottleneck] model: <model_slug>  max_tokens: <MAX_OUTPUT_TOKENS>  strategy: ~<strat_tokens> tokens
```

**If LLM share > 70%**: the bottleneck is the LLM call. The priority list in Phase 2 targets LLM latency. The speed gate uses **token-count reduction** as the proxy for speedup (since LLM wall-clock is proportional to tokens generated).

**If LLM share ≤ 70%**: the bottleneck is the backtest. The priority list targets backtest execution. The speed gate uses **actual `python backtest.py` timing**.

### 0c — Load run history

Check for `results/optimize-loop-history.jsonl`. If it exists, read every line and print a compact summary:

```
[history] Previously kept    : <comma-separated descriptions, or "none">
[history] Previously discarded: <count> attempts — skipping these in proposals
```

List each discarded entry's description and `notes` field (one line each).

Use this history throughout Phase 2 Step 1:
- Skip changes whose description closely matches a prior KEEP.
- Skip changes whose description closely matches a prior DISCARD — UNLESS `notes` explains a fixable reason.

---

## Phase 1 — Establish baseline (runs once, before any changes)

### 1a. Correctness fingerprint

Run `python backtest.py` **once**. Capture stdout. Parse the `---` block and record every metric as the **correctness fingerprint** (used for all iterations regardless of which bottleneck we're targeting):

```
fingerprint = {val_sharpe, sortino, calmar, max_drawdown, win_rate, total_trades, total_return_pct}
```

If the run crashes or produces `val_sharpe: 0.000000` with `total_trades: 0`, stop: "Baseline backtest crashed — fix the strategy before optimizing the harness."

### 1b. Timing baseline

**If bottleneck = LLM** (from Phase 0):
- Run `python backtest.py` once for a rough `T_backtest` (not the optimization target, just for reference).
- Set `T_baseline_tokens = effective_output_tokens` from Phase 0 (this is what we optimize against).
- Print: `[baseline] model=<slug>  T_llm_estimate=<T>s  T_backtest=<T>s  effective_output_tokens=<N>`

**If bottleneck = backtest** (from Phase 0):
- Run `python backtest.py` **five times** back-to-back. Drop min and max. Average remaining three → `T_baseline`.
- Print: `[baseline] val_sharpe=<X>  total_trades=<N>  T_baseline=<T>s (avg of 3 mid runs)`

---

## Phase 2 — Optimization loop (repeat N times)

Label each iteration clearly: `===== optimize-loop iteration I/N =====`

### Step 1 — Propose ONE change

Read the current contents of `loop.py`, `backtest.py`, and `program.md`. Then propose **exactly one** code change targeting the dominant cost.

**Rules:**
- Touch only `loop.py`, `backtest.py`, and/or `program.md` — never `strategies/`, `results/`, `data/`, or `.github/`
- Not alter the mathematical correctness of the backtest
- Be self-contained — one logical idea, not a bundle

---

### Priority list when bottleneck = LLM (most common case)

Work through this list in order; pick the first idea not yet implemented or not already discarded:

**1. Reduce `MAX_OUTPUT_TOKENS`**

In `loop.py`, `MAX_OUTPUT_TOKENS = 8000`. For reasoning models this is the thinking budget — the model burns thinking tokens up to this cap. Reduce it to `max(1500, 3 × strat_tokens)` where `strat_tokens = len(strategy_file) // 4`. This directly limits thinking time.

Example: strategy is ~800 tokens → new cap = max(1500, 2400) = 2400. For a reasoning model at 40 tok/s: 8000 tokens = 200s → 2400 tokens = 60s. **3.3× speedup.**

*Do not reduce below 1500* — the strategy rewrite itself needs room, and too-tight caps cause truncated/malformed output.

**2. Switch to in-process backtest call**

In `loop.py`, `run_backtest()` calls `subprocess.run([PYTHON, "backtest.py"])` — paying ~0.7s of Python cold-start per iteration. Replace with a direct in-process call:

```python
import io, contextlib
import backtest as _bt_module  # already imported at module level in loop.py

def run_backtest() -> dict:
    buf = io.StringIO()
    err_buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err_buf):
        try:
            _bt_module.run()
        except SystemExit:
            pass
    out = buf.getvalue()
    RUN_LOG.write_text(out + "\n----- STDERR -----\n" + err_buf.getvalue(), encoding="utf-8")
    return _parse_summary(out)
```

Note: `backtest` is already imported as `_bt_module` at line ~`import backtest as _bt_module` in `loop.py`. Saves ~0.7s per iteration (small relative to LLM, but free).

**3. Diff-based strategy mutations**

Currently Claude rewrites the **entire** strategy file (~800 tokens). Ask it to output a **unified diff** instead (~50-200 tokens). This is a 4-10× token reduction — the single biggest LLM speedup available.

This requires two coordinated changes in one iteration:
- **`program.md`**: change the output format instruction from "output the complete new strategy file" to "output a unified diff (`--- a/strategy.py`, `+++ b/strategy.py`, `@@ ... @@` format)"
- **`loop.py`**: replace `STRATEGY.write_text(new_code)` with diff-apply logic using `difflib` or `subprocess.run(["patch"])`:

```python
import subprocess, tempfile, os

def _apply_diff(strategy_path: Path, diff_text: str) -> None:
    """Apply a unified diff to the strategy file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
        f.write(diff_text)
        patch_file = f.name
    try:
        result = subprocess.run(
            ["patch", "--no-backup-if-mismatch", str(strategy_path), patch_file],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise ValueError(f"patch failed: {result.stderr}")
    finally:
        os.unlink(patch_file)
```

Update `CODE_RE` regex in loop.py to capture the diff block instead of a Python code block.

This is the highest-leverage change but also the most complex. Implement it only after #1 and #2 have been tried.

**4. Stream the LLM response**

Convert `client.chat.completions.create(...)` to use `stream=True`. Parse the strategy from the accumulated stream. This doesn't reduce total token count but:
- Lets the loop print progress to the dashboard as tokens arrive (better UX)
- Reduces TTFT (time-to-first-token) experience from 3 minutes to seconds

**5. Free-form** — any other single well-reasoned change targeting LLM latency or loop overhead.

---

### Priority list when bottleneck = backtest (rare — only applies if LLM share ≤ 70%)

1. **In-process backtest call** — same as #2 above, saves cold-start overhead
2. **Parallel basket evaluation** — `ThreadPoolExecutor` in `_run_basket()` in `backtest.py`
3. **Eliminate heavy imports** — check for slow module-level imports (e.g. scipy) in `backtest.py`
4. **Parquet read-once caching** — module-level dict keyed by path in `backtest.py`
5. **Free-form**

---

### Step 2 — Snapshot then implement

**Before editing any file**, read the current contents of every file you plan to modify and hold them in memory as `snapshot_<filename>`. This is the safe restore point.

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

If **any field is outside tolerance**, revert and log `DISCARD (correctness gate failed)`.

### Step 4 — Speed gate

**If bottleneck = LLM:**

Compute the estimated output tokens after the change:
- For `MAX_OUTPUT_TOKENS` reduction: `new_tokens = new_MAX_OUTPUT_TOKENS`
- For diff-based mutations: estimate `new_tokens = avg_diff_size_tokens` (count the diff size for the current strategy; a typical one-line change diff is ~50-100 tokens)
- For streaming: `new_tokens = T_baseline_tokens` (no token reduction — gate auto-passes, treat as cost-neutral)
- For in-process backtest: `new_tokens = T_baseline_tokens` (no effect on LLM — evaluate on `T_backtest` timing instead)

```
speedup = T_baseline_tokens / new_tokens
```

Gate passes if `speedup > 1.05`.

**If bottleneck = backtest:**

Run `python backtest.py` **five times**. Drop min and max. Average remaining three → `T_new`.
```
speedup = T_baseline / T_new
```
Gate passes if `speedup > 1.05`.

### Step 5 — Keep or discard

**Keep** (both gates passed):
- Update the baseline (tokens or time) to the new value.
- Log: `KEEP  speedup=<X>x  change="<description>"`
- Do not commit — leave changes in working tree.

**Discard** (any gate failed):
- Revert by writing back the snapshots captured in Step 2.
- Log: `DISCARD (<gate>: <reason>)  change="<description>"`

### Step 6 — Log and summarise

Append one JSON object to `results/optimize-loop-history.jsonl`:

```jsonl
{"date": "YYYY-MM-DD", "description": "<one-line description>", "outcome": "KEEP"|"DISCARD", "gate": "speed"|"correctness"|null, "speedup": <float>, "notes": "<why it failed or why it works>"}
```

Print one line:
```
[I/N] <KEEP|DISCARD>  speedup=<X>x  gate=<correctness|speed|->  change="<one-line description>"
```

---

## Phase 3 — Final report

```
===== /optimize-loop complete =====
Iterations run     : N
Changes kept       : K
Changes dropped    : N-K
Bottleneck targeted: <LLM | backtest>
Total speedup      : <baseline_original / baseline_final>x

Kept changes (review and commit when ready):
  1. <description>  (estimated <X>s/iter saved)
  2. ...

Files modified: <list>

Next steps:
  - Review the diffs: git diff loop.py backtest.py program.md
  - Test end-to-end: python loop.py --iters 1  (requires OPENROUTER_API_KEY)
  - Commit when satisfied: git add -p && git commit -m "perf: <summary>"
  - Run history saved to: results/optimize-loop-history.jsonl
```

---

## Behavior rules

- **One change per iteration.** Diff-based mutations (touching loop.py + program.md) count as one logical change and are allowed.
- **Never touch strategies/, results/, data/, or .github/.** These are always out of scope.
- **Never commit.** The skill edits files but never runs `git add` or `git commit`.
- **Fingerprint is immutable.** Set in Phase 1, never updated.
- **Baseline updates on keep.** After each kept change, update the baseline (token count or timing) so subsequent iterations measure against the current best.
- **Revert on any gate failure.** Always revert before moving on. Never leave a failing change in the working tree.
- **Report clearly.** Every iteration prints its outcome with enough detail to understand what was tried and why it was kept or dropped.
- **Reasoning model warning.** If the model is classified as a reasoning model and `MAX_OUTPUT_TOKENS` is above 3000, print a warning before Phase 2: "⚠ Reasoning model detected with MAX_OUTPUT_TOKENS=N — thinking traces likely dominate iteration time. Reducing max_tokens is the highest-priority optimization."

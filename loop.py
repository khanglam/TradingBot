"""Autoresearch orchestrator — the karpathy-style loop.

The loop ALWAYS runs on the dev branch, never on main. main carries only the
frozen, promoted strategy files that live_trade.py reads; the loop's mutations
and git resets stay on dev until sync_branches.yml merges dev into main.

Each iteration:
  1. Read STRATEGY_FILE + last 10 rows of per-campaign results.tsv + program.md
  2. Ask the LLM for a single mutation (returns description + new strategy file)
  3. Write strategy file, git commit
  4. Compute DSR benchmark from prior trial variance, set DSR_BENCHMARK env
  5. Run backtest.py --window train+val (subprocess), parse summary block
  6. Apply keep/discard rules (in order):
       - both windows: max_dd ≥ MAX_DRAWDOWN_LIMIT  → discard
       - both windows: total_trades < MIN_TRADES    → discard
       - DSR_GATE_THRESHOLD enabled & dsr < gate    → discard
       - val_sub_min < SUB_PERIOD_MIN_SHARPE        → discard (regime gate)
       - val_sub_neg_count > SUB_PERIODS // 2       → discard (regime gate)
       - val_sharpe < best_val_sharpe - VAL_TOLERANCE → discard (holdout gate)
       - OPTIMIZE_METRIC (on train) > best_so_far   → keep
       - else                                        → discard (regression)
       discard / crash → git reset --hard HEAD~1
  7. Append a row to results.tsv (22-col schema; see RESULTS_COLS)
  8. Repeat until --iters reached (or human interrupt; karpathy-style)
  9. Session end: commit accumulated tsv rows in one chore commit.

Requires:
    OPENROUTER_API_KEY in env (or .env file). Get one at https://openrouter.ai/keys
    A clean git working tree on entry (so we can reset cleanly)
    HEAD on the dev branch (set ALLOW_LOOP_ON_MAIN=1 to override)

Environment knobs (all optional):
    OPTIMIZE_METRIC        train_sharpe (default) | train_calmar | train_dsr
    VAL_TOLERANCE          val-Sharpe regression allowed on a keep (default 0.10)
    SUB_PERIOD_MIN_SHARPE  per-period regime floor on val (default -0.5)
    SUB_PERIODS            val sub-period count (default 4; backtest.py reads it)
    WARMUP_BARS            leading-bars purge per window (default 150)
    DSR_GATE_THRESHOLD     reject if dsr below this; 0 disables (default)
    SYMBOLS                comma-sep parquet stems under data/ (e.g.
                           "crypto/BTC_USDT_4h" or "stocks/TSLA_1d,stocks/NVDA_1d").
                           N=1 → single mode; N≥2 → basket mode with overfit penalty.
    OPENROUTER_MODEL       any model slug from openrouter.ai/models. Defaults to
                           deepseek/deepseek-v4-flash. Examples:
                             anthropic/claude-sonnet-4-6
                             openai/gpt-5
                             deepseek/deepseek-r1
                             google/gemini-2.5-pro
    MAX_OUTPUT_TOKENS      LLM reply cap (default 8000). Set in .env.

Usage:
    python loop.py --iters 50
    python loop.py --iters 1   # single experiment, useful for debugging
"""
from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import subprocess
import sys
import time
import datetime
from pathlib import Path

# Windows console defaults to cp1252; force UTF-8 so unicode prints never crash.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).parent
PROGRAM = ROOT / "program.md"
RESULTS_DIR = ROOT / "results"
RUN_LOG = RESULTS_DIR / "run.log"
RESULTS_DIR.mkdir(exist_ok=True)

PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PYTHON).exists():
    PYTHON = sys.executable

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"  # any model slug from openrouter.ai/models
_DEFAULT_MAX_OUTPUT_TOKENS = 8000


def _load_dotenv_files() -> None:
    """Load .env from cwd and repo root."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
    p = ROOT
    for _ in range(5):
        if (p / "configs.toml").exists() and (p / ".env").exists():
            load_dotenv(p / ".env", override=False)
            break
        if p.parent == p:
            break
        p = p.parent


_load_dotenv_files()

import backtest as _bt_module  # noqa: E402

RESULTS = _bt_module.results_path()
STRATEGY = (ROOT / _bt_module.strategy_file()).resolve()
STRATEGY_REL = str(STRATEGY.relative_to(ROOT)).replace("\\", "/")


def _max_output_tokens() -> int:
    """From MAX_OUTPUT_TOKENS in .env (loaded by _load_dotenv_files before loop runs)."""
    return max(1500, int(os.environ.get("MAX_OUTPUT_TOKENS", str(_DEFAULT_MAX_OUTPUT_TOKENS))))


KEEP_THRESHOLD = float(os.environ.get("KEEP_THRESHOLD", "0.0"))  # strictly > best
MAX_DRAWDOWN_LIMIT = float(_bt_module._cfg("MAX_DRAWDOWN_LIMIT", "30.0"))
MIN_TRADES = int(_bt_module._cfg("MIN_TRADES", "20"))
# Karpathy autoresearch runs until --iters or human interrupt — no strike-out.
# Failure is the steady state at ~25% keep rate. Set MAX_REGRESSIONS>0 in env
# to re-enable a brake for unattended runs.
MAX_REGRESSIONS = int(os.environ.get("MAX_REGRESSIONS", 0))

# Annualization factor — must match backtest.py's ANN_FACTOR_4H so we can
# convert logged sharpe_ann_4h values back to per-bar units for DSR variance.
ANN_FACTOR_4H = math.sqrt(365 * 6)

# OPTIMIZE_METRIC env var picks the keep/discard gradient — computed on the
# TRAIN window. val is treated as a non-degradation gate, not an objective.
# This is the holdout discipline fix: with thousands of LLM-driven iterations,
# optimizing directly on val turns val into a de-facto training set. Train is
# now the surface we descend; val is the brake.
# Allowed (each suffixed with the train_ window):
#   train_sharpe (default) — backtesting.py reported Sharpe on train
#   train_calmar          — train return / train drawdown
#   train_dsr             — Deflated Sharpe on train (use only at N ≥ 50 trials)
# All metrics are logged to results.tsv regardless; this just picks the gradient.
ALLOWED_METRICS = {"train_sharpe", "train_calmar", "train_dsr"}
OPTIMIZE_METRIC = os.environ.get("OPTIMIZE_METRIC", "train_sharpe")
if OPTIMIZE_METRIC not in ALLOWED_METRICS:
    raise SystemExit(
        f"OPTIMIZE_METRIC must be one of {sorted(ALLOWED_METRICS)}, got {OPTIMIZE_METRIC!r}"
    )

# How much the val-window Sharpe is allowed to drop on a "kept" train improvement.
# Set to 0 to require strict val improvement (textbook holdout); the default 0.10
# leaves room for honest variance while still vetoing trades that improve train
# at val's expense.
try:
    VAL_TOLERANCE = float(os.environ.get("VAL_TOLERANCE", "0.10") or 0.10)
except ValueError:
    VAL_TOLERANCE = 0.10

# Per-period regime-stability gates (P2).
# SUB_PERIOD_MIN_SHARPE: floor for the worst sub-period of val. Below it, the
# strategy is regime-fragile and gets rejected regardless of overall Sharpe.
try:
    SUB_PERIOD_MIN_SHARPE = float(os.environ.get("SUB_PERIOD_MIN_SHARPE", "-0.5") or -0.5)
except ValueError:
    SUB_PERIOD_MIN_SHARPE = -0.5

# DSR_GATE_THRESHOLD: if > 0, any trial with dsr below this is discarded
# regardless of OPTIMIZE_METRIC improvement. Disabled (0) by default —
# enable it (e.g. 0.5) once you have 50+ non-crash trials. Below that, DSR's
# variance estimator is too noisy to gate honestly.
try:
    DSR_GATE_THRESHOLD = float(os.environ.get("DSR_GATE_THRESHOLD", "0") or 0.0)
except ValueError:
    DSR_GATE_THRESHOLD = 0.0


# ──────────────────────────── git helpers ──────────────────────────────

def _git(*args: str) -> str:
    res = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return res.stdout.strip()


def git_dirty() -> bool:
    return bool(_git("status", "--porcelain"))


def git_current_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD")


def assert_dev_branch() -> None:
    """The loop mutates and resets HEAD; it must run on dev, never on main.
    Set ALLOW_LOOP_ON_MAIN=1 to override for one-off local experiments."""
    if os.environ.get("ALLOW_LOOP_ON_MAIN") == "1":
        return
    branch = git_current_branch()
    if branch != "dev":
        raise SystemExit(
            f"ERROR: loop.py refuses to run on branch {branch!r}. "
            f"Switch to dev (`git checkout dev`) or set ALLOW_LOOP_ON_MAIN=1 to override."
        )


def git_commit_results() -> bool:
    """Stage and commit any pending experiment rows in results/*.tsv.

    Called at the start AND end of every loop session so the tsv never sits
    dirty in the working tree:
      - start: catches rows from a prior aborted session before the dirty-check
      - end:   commits this session's accumulated rows in one chore commit

    Pathspec-scoped to `results/` only. If there are other staged changes
    in the index, this function does NOT sweep them into the chore commit
    — they stay staged for whoever staged them.

    Returns True if a commit was created, False if nothing to commit."""
    # Stage tsv changes (only). `git add -- results/*.tsv` adds tracked tsv
    # files that have been modified.
    _git("add", "--", "results/")
    # Diff cached, scoped to results/. If empty, there's nothing for us
    # specifically to do.
    diff = _git("diff", "--cached", "--name-only", "--", "results/")
    if not diff.strip():
        return False
    # `git commit -- <pathspec>` filters the staged set by pathspec, so any
    # other staged changes from an outer workflow stay staged.
    _git("commit", "-m", "experiment: checkpoint experiment log",
         "--no-verify", "--", "results/")
    print("[loop] committed pending results.tsv rows")
    return True


def git_short_sha() -> str:
    return _git("rev-parse", "--short=7", "HEAD")


def git_commit_strategy(description: str) -> str | None:
    """Commit the active strategy file. Returns None if the LLM returned
    bytes-identical content (no diff) — the caller must treat this as
    a no-op iteration, not a crash."""
    _git("add", str(STRATEGY))
    diff = _git("diff", "--cached", "--name-only")
    if not diff.strip():
        return None
    _git("commit", "-m", f"experiment ({STRATEGY_REL}): {description}", "--no-verify")
    return git_short_sha()


def git_reset_last() -> None:
    """Revert the last strategy commit. results.tsv is tracked, so a bare
    --hard would also wipe rows accumulated this session — preserve it."""
    saved = RESULTS.read_text(encoding="utf-8") if RESULTS.exists() else None
    _git("reset", "--hard", "HEAD~1")
    if saved is not None:
        RESULTS.write_text(saved, encoding="utf-8")


# ──────────────────────────── results.tsv ──────────────────────────────

# Schema columns in fixed order. Migrations below pad missing columns with
# empty strings — schema growth is supported, but never reorder existing
# columns or rename them.
RESULTS_COLS = [
    "commit", "val_sharpe", "sortino", "sharpe_ann_4h", "calmar", "psr", "dsr",
    "skew", "kurtosis", "max_drawdown", "win_rate", "total_trades",
    "train_sharpe", "train_calmar", "train_max_drawdown", "train_total_trades",
    "val_sub_min", "val_sub_std", "val_sub_neg_count",
    "status", "timestamp", "description",
]
RESULTS_HEADER = "\t".join(RESULTS_COLS) + "\n"

# Historical schemas, oldest first.
LEGACY_SCHEMAS = [
    # v0 — original 7-column layout
    [
        "commit", "val_sharpe", "max_drawdown", "win_rate", "total_trades",
        "status", "description",
    ],
    # v1 — added sortino/sharpe_ann_4h/calmar/psr/skew/kurtosis (no dsr)
    [
        "commit", "val_sharpe", "sortino", "sharpe_ann_4h", "calmar", "psr",
        "skew", "kurtosis", "max_drawdown", "win_rate", "total_trades",
        "status", "description",
    ],
    # v2 - added dsr
    [
        "commit", "val_sharpe", "sortino", "sharpe_ann_4h", "calmar", "psr", "dsr",
        "skew", "kurtosis", "max_drawdown", "win_rate", "total_trades",
        "status", "description",
    ],
    # v3 — added timestamp (current pre-train-window schema)
    [
        "commit", "val_sharpe", "sortino", "sharpe_ann_4h", "calmar", "psr", "dsr",
        "skew", "kurtosis", "max_drawdown", "win_rate", "total_trades",
        "status", "timestamp", "description",
    ],
]


def _migrate_row(parts: list[str], old_cols: list[str]) -> list[str]:
    """Map a single row from an old schema into the current schema. Missing
    columns become empty strings."""
    old_map = dict(zip(old_cols, parts))
    return [old_map.get(c, "") for c in RESULTS_COLS]


def _ensure_results() -> None:
    """Create results.tsv with current schema, or migrate any known legacy
    layout in place. Backs up the original file before rewriting."""
    if not RESULTS.exists():
        RESULTS.write_text(RESULTS_HEADER, encoding="utf-8")
        return

    lines = RESULTS.read_text(encoding="utf-8").splitlines()
    if not lines:
        RESULTS.write_text(RESULTS_HEADER, encoding="utf-8")
        return

    header_cols = lines[0].split("\t")
    if header_cols == RESULTS_COLS:
        return  # current schema, no migration needed

    matched = next((cols for cols in LEGACY_SCHEMAS if header_cols == cols), None)
    if matched is None:
        raise RuntimeError(
            f"results.tsv has unrecognized header: {header_cols}. "
            f"Expected one of: current={RESULTS_COLS} or legacy={LEGACY_SCHEMAS}."
        )

    backup = RESULTS.with_suffix(".tsv.legacy")
    if not backup.exists():
        backup.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[loop] migrated {RESULTS.name} (legacy backup at {backup.name})")

    new_lines = [RESULTS_HEADER.rstrip("\n")]
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < len(matched):
            parts = parts + [""] * (len(matched) - len(parts))
        new_lines.append("\t".join(_migrate_row(parts, matched)))
    RESULTS.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def append_result(sha: str, metrics: dict, status: str, description: str) -> None:
    """Append a row using the full new schema. metrics dict comes from
    run_backtest() and is expected to contain every numeric column."""
    _ensure_results()
    row = [
        sha,
        f"{metrics.get('val_sharpe', 0.0):.6f}",
        f"{metrics.get('sortino', 0.0):.6f}",
        f"{metrics.get('sharpe_ann_4h', 0.0):.6f}",
        f"{metrics.get('calmar', 0.0):.6f}",
        f"{metrics.get('psr', 0.0):.6f}",
        f"{metrics.get('dsr', 0.0):.6f}",
        f"{metrics.get('skew', 0.0):.3f}",
        f"{metrics.get('kurtosis', 0.0):.3f}",
        f"{metrics.get('max_drawdown', 0.0):.2f}",
        f"{metrics.get('win_rate', 0.0):.3f}",
        f"{metrics.get('total_trades', 0)}",
        f"{metrics.get('train_sharpe', 0.0):.6f}",
        f"{metrics.get('train_calmar', 0.0):.6f}",
        f"{metrics.get('train_max_drawdown', 0.0):.2f}",
        f"{metrics.get('train_total_trades', 0)}",
        f"{metrics.get('val_sub_min', 0.0):.6f}",
        f"{metrics.get('val_sub_std', 0.0):.6f}",
        f"{metrics.get('val_sub_neg_count', 0)}",
        status,
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        description,
    ]
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write("\t".join(row) + "\n")


def best_metric_so_far(metric: str) -> float:
    """Best value of `metric` across kept rows. Reads by header column name
    so it works on both legacy-migrated files and new files."""
    if not RESULTS.exists():
        return 0.0
    lines = RESULTS.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return 0.0
    header = lines[0].split("\t")
    try:
        idx_metric = header.index(metric)
        idx_status = header.index("status")
    except ValueError:
        return 0.0
    best = 0.0
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= max(idx_metric, idx_status):
            continue
        if parts[idx_status] != "keep":
            continue
        try:
            best = max(best, float(parts[idx_metric]))
        except ValueError:
            continue
    return best


def last_n_rows(n: int = 10) -> list[dict]:
    if not RESULTS.exists():
        return []
    lines = RESULTS.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    out = []
    for line in lines[-n:]:
        parts = line.split("\t")
        if len(parts) < len(header):
            parts = parts + [""] * (len(header) - len(parts))
        out.append(dict(zip(header, parts)))
    return out


# ─────────────────────────── backtest runner ───────────────────────────

SUMMARY_FIELDS = {
    "val_sharpe": float, "sortino": float, "sharpe_ann_4h": float,
    "calmar": float, "psr": float, "dsr": float, "skew": float, "kurtosis": float,
    "max_drawdown": float, "win_rate": float, "total_trades": int,
    "total_return_pct": float,
    # P1 — train window companion metrics (loop optimizes on train_sharpe and
    # uses val_sharpe as a non-degradation gate).
    "train_sharpe": float, "train_calmar": float,
    "train_max_drawdown": float, "train_total_trades": int,
    # P2 — per-period regime-stability metrics computed on val.
    "val_sub_min": float, "val_sub_std": float, "val_sub_neg_count": int,
}


def _parse_summary(out: str) -> dict:
    """Parse `key: value` lines between two `---` markers. Tolerant of any
    field order; missing fields default to 0."""
    metrics: dict[str, float | int] = {k: (0 if t is int else 0.0) for k, t in SUMMARY_FIELDS.items()}
    in_block = False
    for raw in out.splitlines():
        line = raw.strip()
        if line == "---":
            in_block = not in_block
            continue
        if not in_block:
            continue
        m = re.match(r"^([a-z_0-9]+):\s+([-\d.]+)\s*$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if key in SUMMARY_FIELDS:
            try:
                metrics[key] = SUMMARY_FIELDS[key](val)
            except ValueError:
                pass
    return metrics


def run_backtest() -> dict:
    """Run backtest.py in a subprocess, parse summary block, return metrics.
    Inherits DSR_BENCHMARK, SYMBOLS, and BASKET_PENALTY env vars from the caller.

    Always uses --window train+val so the loop has both Sharpes to gate on:
    train is the optimization signal; val is the anti-overfit selector.
    """
    proc = subprocess.run(
        [PYTHON, "backtest.py", "--window", "train+val"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = proc.stdout
    RUN_LOG.write_text(out + "\n----- STDERR -----\n" + proc.stderr, encoding="utf-8")
    return _parse_summary(out)


# ──────────────────────── crash feed-forward ─────────────────────────

# When a backtest crashes, the loop reverts the offending strategy.py via
# git_reset_last(). The next iteration's LLM call would otherwise see
# only `status=crash` in the history row — no traceback, no idea what broke.
# We rebuild the crash context on demand at call_llm time:
#   - source: git_show on the crashed SHA from results.tsv
#       (commit object survives the reset; lives in reflog + as a loose
#        object until git-gc, ~90d default — plenty of time for one tick)
#   - stderr: tail of run.log
#       (overwritten only inside run_backtest(), which runs after call_llm
#        each iteration, so at call_llm time it still holds prior stderr)


def _read_stderr_tail() -> str:
    """Pull the stderr block written by run_backtest() out of run.log."""
    if not RUN_LOG.exists():
        return ""
    try:
        content = RUN_LOG.read_text(encoding="utf-8")
    except Exception:
        return ""
    parts = content.split("----- STDERR -----")
    return parts[-1].strip() if len(parts) > 1 else ""


# ──────────────────────────── DSR benchmark ────────────────────────────

def compute_dsr_benchmark() -> float:
    """Compute the DSR benchmark in *per-bar* Sharpe units from prior trial
    history. Bailey & López de Prado's DSR adjusts the observed in-sample
    Sharpe for the fact that running N trials inflates the maximum by chance.

    SR* = √V[SR̂_n] × E[max_n]   (per-bar)

    where V[SR̂_n] is the variance of trial Sharpes (per-bar) and E[max_n] is
    the expected maximum of N standard-normal draws — approximated here by
    √(2·log(N)) which is the leading term of the Gumbel envelope.

    Returns 0 if there isn't enough history to compute a meaningful benchmark
    (which makes DSR collapse to PSR — safe default at small N).
    """
    if not RESULTS.exists():
        return 0.0
    lines = RESULTS.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return 0.0
    header = lines[0].split("\t")
    try:
        idx_sa = header.index("sharpe_ann_4h")
        idx_status = header.index("status")
    except ValueError:
        return 0.0

    sharpes_per_bar: list[float] = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= max(idx_sa, idx_status):
            continue
        if parts[idx_status] == "crash":
            continue
        try:
            sa = float(parts[idx_sa])
        except ValueError:
            continue
        if sa == 0.0:
            continue  # missing value (legacy row) or genuine zero — exclude either way
        sharpes_per_bar.append(sa / ANN_FACTOR_4H)

    n = len(sharpes_per_bar)
    if n < 2:
        return 0.0
    var_sharpe = statistics.pvariance(sharpes_per_bar)
    if var_sharpe <= 0:
        return 0.0
    expected_max = math.sqrt(2.0 * math.log(n))
    return math.sqrt(var_sharpe) * expected_max


# ──────────────────────────── LLM call (OpenRouter) ──────────────────────────────

DESCRIPTION_RE = re.compile(r"##\s*Description\s*\n+(.+?)(?=\n##|\Z)", re.DOTALL)
# Accept any `*.py` filename in the section header — `## strategy.py`,
# `## strategies/stocks.py`, `## strategies/crypto.py`, etc. — so the
# section label matches whatever file is actively being edited.
CODE_RE = re.compile(r"##\s*[\w/]+\.py\s*\n+```python\s*\n(.*?)\n```", re.DOTALL)


def call_llm(strategy_src: str, program_src: str, history: list[dict]) -> tuple[str, str]:
    """Returns (description, new_strategy_source).

    Uses OpenRouter (https://openrouter.ai) as the LLM backend. Any model
    available on OpenRouter can be selected via OPENROUTER_MODEL in .env.

    If the most recent history row is a crash, the crashed strategy.py source
    (recovered from git via the row's commit SHA) and the traceback (tail of
    run.log) are injected into the prompt so the LLM can fix the actual bug
    instead of guessing from `status=crash`."""
    import httpx
    import openai

    _http_client = httpx.Client(verify=False)

    client = openai.OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
        http_client=_http_client,
    )

    history_block = "(no prior experiments)"
    if history:
        rows = ["commit\tsharpe\tmax_dd\ttrades\tstatus\tdescription"]
        for r in history:
            rows.append(
                f"{r['commit']}\t{r['val_sharpe']}\t{r['max_drawdown']}\t"
                f"{r['total_trades']}\t{r['status']}\t{r['description']}"
            )
        history_block = "\n".join(rows)

    user_msg = (
        f"# Current {STRATEGY_REL}\n```python\n{strategy_src}\n```\n\n"
        f"# Last {len(history)} experiments (results.tsv tail)\n```\n{history_block}\n```\n\n"
    )

    # If the immediately-previous iteration crashed, show the LLM the exact
    # code that broke and the traceback. Without this, the loop wastes
    # iterations re-proposing the same broken pattern.
    # However, if the LLM repeatedly fails to fix the crash (e.g., due to laziness),
    # we stop feeding the crash back after 3 attempts to break the infinite loop.
    consecutive_crashes = 0
    for r in reversed(history):
        if r.get("status") == "crash":
            consecutive_crashes += 1
        else:
            break

    if consecutive_crashes > 0 and consecutive_crashes <= 3:
        sha = history[-1].get("commit", "")
        crashed_src = ""
        if sha:
            # Try the active campaign's strategy file first; fall back to the
            # legacy "strategy.py" path for crashes recorded before the split.
            for candidate in (STRATEGY_REL, "strategy.py"):
                try:
                    crashed_src = _git("show", f"{sha}:{candidate}")
                    break
                except subprocess.CalledProcessError:
                    continue
        if not crashed_src:
            crashed_src = "(commit not retrievable — likely git-gc'd)"
        crashed_stderr = _read_stderr_tail()[-2000:]
        user_msg += (
            "# CRITICAL CONTEXT: YOUR LAST EXPERIMENT CRASHED!\n"
            f"Below is the {STRATEGY_REL} that broke and the Python traceback.\n"
            "Your primary goal right now is to diagnose and fix this error — "
            "do not propose new features until the bug is gone. "
            "If the traceback is empty, the crash was a 0-trade run "
            "(filters too strict): relax or remove a condition.\n\n"
            f"## Crashed {STRATEGY_REL} (commit {sha or '?'})\n"
            f"```python\n{crashed_src}\n```\n\n"
            f"## Traceback / stderr (tail)\n```text\n{crashed_stderr}\n```\n\n"
        )

    user_msg += "Propose ONE change. Reply with the two required sections only."

    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=_max_output_tokens(),
        messages=[
            {"role": "system", "content": program_src},
            {"role": "user", "content": user_msg},
        ],
    )
    text = resp.choices[0].message.content or ""

    desc_m = DESCRIPTION_RE.search(text)
    code_m = CODE_RE.search(text)
    if not desc_m or not code_m:
        raise ValueError(f"could not parse LLM response:\n{text[:1000]}")

    description = desc_m.group(1).strip().splitlines()[0][:200]
    new_code = code_m.group(1)
    return description, new_code


# ──────────────────────────────── main ─────────────────────────────────

def one_iteration(best_metric: float, best_val_sharpe: float) -> tuple[str, float, float]:
    """Run one experiment. Returns (status, new_best_train_metric, new_best_val_sharpe).

    Optimization signal: OPTIMIZE_METRIC on TRAIN (default train_sharpe).
    Anti-overfit gates (a candidate must pass ALL):
      - val_sharpe ≥ best_val_sharpe - VAL_TOLERANCE     (val brake)
      - val_sub_min ≥ SUB_PERIOD_MIN_SHARPE              (regime brake)
      - val_sub_neg_count ≤ SUB_PERIODS // 2             (regime brake)
      - max_drawdown on BOTH windows < MAX_DRAWDOWN_LIMIT
      - total_trades on BOTH windows ≥ MIN_TRADES
      - existing DSR multiple-testing gate (on the optimize-side dsr)
    """
    strategy_src = STRATEGY.read_text(encoding="utf-8")
    program_src = PROGRAM.read_text(encoding="utf-8")
    history = last_n_rows(10)

    if history and history[-1].get("status") == "crash":
        print(f"[loop] feeding back last crash (commit {history[-1].get('commit', '?')}) to LLM")

    print(
        f"\n[loop] optimize={OPTIMIZE_METRIC}  best={best_metric:.4f}  "
        f"best_val_sharpe={best_val_sharpe:.4f}  asking LLM…"
    )
    try:
        description, new_code = call_llm(strategy_src, program_src, history)
    except Exception as e:
        print(f"[loop] llm error: {e}")
        return "llm_error", best_metric, best_val_sharpe

    print(f"[loop] proposed: {description}")
    STRATEGY.write_text(new_code, encoding="utf-8")
    sha = git_commit_strategy(description)
    if sha is None:
        # LLM returned bytes-identical strategy.py — degenerate mutation.
        # No commit was made; nothing to revert. Skip the backtest entirely.
        print("[loop] no-change → skipping (LLM returned identical strategy)")
        return "noop", best_metric, best_val_sharpe
    print(f"[loop] committed {sha}, running backtest…")

    # Set DSR benchmark from prior-trial variance before invoking backtest.py.
    # Backtest reads DSR_BENCHMARK and emits a deflated PSR ("dsr").
    dsr_benchmark = compute_dsr_benchmark()
    os.environ["DSR_BENCHMARK"] = f"{dsr_benchmark:.10f}"
    if dsr_benchmark > 0:
        print(f"[loop] DSR benchmark (per-bar SR*) = {dsr_benchmark:.6f}")

    metrics = run_backtest()
    val_sharpe = metrics["val_sharpe"]
    calmar = metrics["calmar"]
    dsr = metrics.get("dsr", 0.0)
    val_max_dd = metrics["max_drawdown"]
    val_trades = metrics["total_trades"]
    win_rate = metrics["win_rate"]
    train_sharpe = metrics.get("train_sharpe", 0.0)
    train_max_dd = metrics.get("train_max_drawdown", 0.0)
    train_trades = int(metrics.get("train_total_trades", 0))
    sub_min = metrics.get("val_sub_min", 0.0)
    sub_neg = int(metrics.get("val_sub_neg_count", 0))

    score = float(metrics.get(OPTIMIZE_METRIC, 0.0))

    print(
        f"[loop] train: sharpe={train_sharpe:.4f}  dd={train_max_dd:.2f}%  trades={train_trades}\n"
        f"[loop] val:   sharpe={val_sharpe:.4f}  dd={val_max_dd:.2f}%  trades={val_trades}  "
        f"win_rate={win_rate:.3f}\n"
        f"[loop] sub:   min={sub_min:.4f}  neg={sub_neg}/{_bt_module.SUB_PERIODS}  "
        f"calmar={calmar:.4f}  dsr={dsr:.4f}\n"
        f"[loop] → {OPTIMIZE_METRIC}={score:.4f}"
    )

    sub_neg_limit = _bt_module.SUB_PERIODS // 2

    # Crash detection: both windows zero-Sharpe AND zero-trades signals an
    # actual failure (vs. constraint violation, which would have a nonzero
    # value somewhere).
    if train_sharpe == 0.0 and val_sharpe == 0.0 and train_trades == 0 and val_trades == 0:
        status = "crash"
    elif val_max_dd >= MAX_DRAWDOWN_LIMIT or train_max_dd >= MAX_DRAWDOWN_LIMIT:
        status = "discard"
        print(f"[loop] dd gate: train={train_max_dd:.2f}% val={val_max_dd:.2f}% limit={MAX_DRAWDOWN_LIMIT}")
    elif val_trades < MIN_TRADES or train_trades < MIN_TRADES:
        status = "discard"
        print(f"[loop] trade-count gate: train={train_trades} val={val_trades} min={MIN_TRADES}")
    elif DSR_GATE_THRESHOLD > 0 and dsr < DSR_GATE_THRESHOLD:
        # Multiple-testing gate: reject "lucky" candidates that pass raw Sharpe
        # but fail the trial-count-adjusted significance check.
        status = "discard"
        print(f"[loop] dsr gate: {dsr:.4f} < {DSR_GATE_THRESHOLD:.4f} → reject")
    elif sub_min < SUB_PERIOD_MIN_SHARPE:
        # P2 regime gate: one of the val sub-periods is too negative —
        # strategy is regime-fragile, reject even if aggregate looks fine.
        status = "discard"
        print(f"[loop] regime gate: val_sub_min={sub_min:.4f} < {SUB_PERIOD_MIN_SHARPE:.4f}")
    elif sub_neg > sub_neg_limit:
        # P2 regime gate: more than half of val sub-periods are losing.
        status = "discard"
        print(f"[loop] regime gate: val_sub_neg_count={sub_neg} > {sub_neg_limit}")
    elif val_sharpe < best_val_sharpe - VAL_TOLERANCE:
        # P1 val-degradation gate: train can improve all it wants, but val
        # is the canary. If val drops more than VAL_TOLERANCE relative to
        # the best val we've seen on a kept commit, we're overfitting train.
        status = "discard"
        print(
            f"[loop] val-degradation gate: val_sharpe={val_sharpe:.4f} < "
            f"best_val={best_val_sharpe:.4f} - tol={VAL_TOLERANCE:.4f}"
        )
    elif score > best_metric + KEEP_THRESHOLD:
        status = "keep"
    else:
        status = "discard"

    if status != "keep":
        git_reset_last()
        print(f"[loop] {status} → reverted")
    else:
        best_metric = score
        # Only ratchet val high-water if we kept the commit. A discarded
        # candidate's val_sharpe doesn't get to set the bar for future runs.
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
        print(f"[loop] KEEP — new best {OPTIMIZE_METRIC}={score:.4f}  val_sharpe={val_sharpe:.4f}")

    append_result(sha, metrics, status, description)
    return status, best_metric, best_val_sharpe


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "ERROR: OPENROUTER_API_KEY not set. Get a key at https://openrouter.ai/keys "
            "and add it to .env (see .env.example).",
            file=sys.stderr,
        )
        return 2

    # Refuse to mutate main — loop runs on dev only.
    assert_dev_branch()

    # Layer 1 — checkpoint anything left dirty by a prior aborted session
    # so the dirty-check below sees a clean tree.
    git_commit_results()

    if git_dirty():
        print("ERROR: git working tree is dirty. Commit or stash first.", file=sys.stderr)
        return 2

    _ensure_results()
    best = best_metric_so_far(OPTIMIZE_METRIC)
    best_val_sharpe = best_metric_so_far("val_sharpe")
    gate_msg = f"  DSR_GATE={DSR_GATE_THRESHOLD:.2f}" if DSR_GATE_THRESHOLD > 0 else ""
    print(
        f"[loop] campaign={STRATEGY_REL}  symbols={_bt_module._cfg('SYMBOLS', '')}\n"
        f"[loop] results={RESULTS.relative_to(ROOT)}  "
        f"train={_bt_module._cfg('TRAIN_START', '')}→{_bt_module._cfg('TRAIN_END', '')}  "
        f"val={_bt_module._cfg('VAL_START', '')}→{_bt_module._cfg('VAL_END', '')}\n"
        f"[loop] OPTIMIZE_METRIC={OPTIMIZE_METRIC}  starting best={best:.4f}  "
        f"best_val_sharpe={best_val_sharpe:.4f}  VAL_TOLERANCE={VAL_TOLERANCE:.2f}{gate_msg}"
    )
    consecutive_regressions = 0

    # Layer 2 — try/finally guarantees we commit accumulated tsv rows even if
    # the user Ctrl-C's mid-loop or an iteration crashes uncaught. The CI
    # workflow's separate commit step is now redundant.
    rc = 0
    try:
        for i in range(1, args.iters + 1):
            print(f"\n========== iteration {i}/{args.iters} ==========")
            status, best, best_val_sharpe = one_iteration(best, best_val_sharpe)

            if status == "keep":
                consecutive_regressions = 0
            elif status == "llm_error":
                time.sleep(10)
            elif status == "noop":
                # No real attempt made (identical code) — don't count toward streak.
                pass
            else:
                consecutive_regressions += 1
                if MAX_REGRESSIONS > 0 and consecutive_regressions >= MAX_REGRESSIONS:
                    print(f"\n[loop] {MAX_REGRESSIONS} consecutive regressions — freezing for review.")
                    rc = 1
                    break
    finally:
        git_commit_results()
    return rc


if __name__ == "__main__":
    sys.exit(main())

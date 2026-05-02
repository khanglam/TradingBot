"""Autoresearch orchestrator — the karpathy-style loop.

Each iteration:
  1. Read STRATEGY_FILE + last 10 rows of per-campaign results.tsv + program.md
  2. Ask Claude for a single mutation (returns description + new strategy file)
  3. Write strategy file, git commit
  4. Compute DSR benchmark from prior trial variance, set DSR_BENCHMARK env
  5. Run backtest.py (subprocess), parse summary block
  6. Apply keep/discard rules:
       - constraints fail (max_dd, min_trades) → discard
       - DSR_GATE_THRESHOLD enabled and dsr < threshold → discard
       - OPTIMIZE_METRIC > best_so_far → keep
       - else → discard (regression)
       discard / crash → git reset --hard HEAD~1
  7. Append a row to results.tsv
  8. Repeat until --iters reached (or human interrupt; karpathy-style)
  9. Session end: commit accumulated tsv rows in one chore commit.

Requires:
    OPENROUTER_API_KEY in env (or .env file). Get one at https://openrouter.ai/keys
    A clean git working tree on entry (so we can reset cleanly)

Environment knobs (all optional):
    OPTIMIZE_METRIC        val_sharpe (default) | calmar | dsr
    DSR_GATE_THRESHOLD     reject if dsr below this; 0 disables (default)
    SYMBOLS                comma-sep parquet stems under data/ (e.g.
                           "crypto/BTC_USDT_4h" or "stocks/TSLA_1d,stocks/NVDA_1d").
                           N=1 → single mode; N≥2 → basket mode with overfit penalty.
    OPENROUTER_MODEL       any model slug from openrouter.ai/models. Defaults to
                           anthropic/claude-haiku-4-5. Examples:
                             anthropic/claude-sonnet-4-6
                             openai/gpt-5
                             deepseek/deepseek-r1
                             google/gemini-2.5-pro

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

# RESULTS path is per-campaign — derived from active SYMBOLS × val-window so
# each (asset, window) combo gets its own history. Switching SYMBOLS env var
# does not clobber prior research; both files coexist under results/.
#
# STRATEGY is the per-campaign strategy file the loop reads/writes/commits.
# Defaults derived from SYMBOLS prefix (crypto/* → strategies/crypto.py,
# else → strategies/stocks.py). Override with STRATEGY_FILE env var.
import backtest as _bt_module
RESULTS = _bt_module.results_path()
STRATEGY = (ROOT / _bt_module.STRATEGY_FILE).resolve()
STRATEGY_REL = str(STRATEGY.relative_to(ROOT)).replace("\\", "/")

PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PYTHON).exists():
    PYTHON = sys.executable

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-haiku-4-5"  # any model slug from openrouter.ai/models
MAX_OUTPUT_TOKENS = 8000

KEEP_THRESHOLD = float(os.environ.get("KEEP_THRESHOLD", "0.0"))  # strictly > best
MAX_DRAWDOWN_LIMIT = float(os.environ.get("MAX_DRAWDOWN_LIMIT", "30.0"))
MIN_TRADES = int(os.environ.get("MIN_TRADES", "20"))
# Karpathy autoresearch runs until --iters or human interrupt — no strike-out.
# Failure is the steady state at ~25% keep rate. Set MAX_REGRESSIONS>0 in env
# to re-enable a brake for unattended runs.
MAX_REGRESSIONS = int(os.environ.get("MAX_REGRESSIONS", 0))

# Annualization factor — must match backtest.py's ANN_FACTOR_4H so we can
# convert logged sharpe_ann_4h values back to per-bar units for DSR variance.
ANN_FACTOR_4H = math.sqrt(365 * 6)

# OPTIMIZE_METRIC env var picks the keep/discard scalar.
# Allowed:
#   val_sharpe (default; backtesting.py reported Sharpe)
#   calmar     (total_return / max_drawdown — return-aware)
#   dsr        (Deflated Sharpe Ratio — multiple-testing-corrected; not
#               recommended as primary metric until N ≥ 50 non-crash trials,
#               since DSR is unstable at small N)
# All metrics are logged to results.tsv regardless; this just picks the gradient.
ALLOWED_METRICS = {"val_sharpe", "calmar", "dsr"}
OPTIMIZE_METRIC = os.environ.get("OPTIMIZE_METRIC", "val_sharpe")
if OPTIMIZE_METRIC not in ALLOWED_METRICS:
    raise SystemExit(
        f"OPTIMIZE_METRIC must be one of {sorted(ALLOWED_METRICS)}, got {OPTIMIZE_METRIC!r}"
    )

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
    _git("commit", "-m", "chore: checkpoint experiment log",
         "--no-verify", "--", "results/")
    print("[loop] committed pending results.tsv rows")
    return True


def git_short_sha() -> str:
    return _git("rev-parse", "--short=7", "HEAD")


def git_commit_strategy(description: str) -> str | None:
    """Commit the active strategy file. Returns None if Claude wrote
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
    "status", "description",
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
        status,
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
    Inherits DSR_BENCHMARK, SYMBOLS, and BASKET_PENALTY env vars from the caller."""
    proc = subprocess.run(
        [PYTHON, "backtest.py"],
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
# git_reset_last(). The next iteration's Claude call would otherwise see
# only `status=crash` in the history row — no traceback, no idea what broke.
# We rebuild the crash context on demand at call_claude time:
#   - source: git_show on the crashed SHA from results.tsv
#       (commit object survives the reset; lives in reflog + as a loose
#        object until git-gc, ~90d default — plenty of time for one tick)
#   - stderr: tail of run.log
#       (overwritten only inside run_backtest(), which runs after call_claude
#        each iteration, so at call_claude time it still holds prior stderr)


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


def call_claude(strategy_src: str, program_src: str, history: list[dict]) -> tuple[str, str]:
    """Returns (description, new_strategy_source).

    Uses OpenRouter (https://openrouter.ai) as the LLM backend. Any model
    available on OpenRouter can be selected via OPENROUTER_MODEL in .env.

    If the most recent history row is a crash, the crashed strategy.py source
    (recovered from git via the row's commit SHA) and the traceback (tail of
    run.log) are injected into the prompt so the LLM can fix the actual bug
    instead of guessing from `status=crash`."""
    import openai

    client = openai.OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
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
    # we stop feeding the crash back after 2 attempts to break the infinite loop.
    consecutive_crashes = 0
    for r in reversed(history):
        if r.get("status") == "crash":
            consecutive_crashes += 1
        else:
            break

    if consecutive_crashes > 0 and consecutive_crashes <= 2:
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
        max_tokens=MAX_OUTPUT_TOKENS,
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

def one_iteration(best_metric: float) -> tuple[str, float]:
    """Run one experiment. Returns (status, new_best_for_active_metric)."""
    strategy_src = STRATEGY.read_text(encoding="utf-8")
    program_src = PROGRAM.read_text(encoding="utf-8")
    history = last_n_rows(10)

    if history and history[-1].get("status") == "crash":
        print(f"[loop] feeding back last crash (commit {history[-1].get('commit', '?')}) to Claude")

    print(f"\n[loop] optimize={OPTIMIZE_METRIC}  best_so_far={best_metric:.4f}  asking Claude…")
    try:
        description, new_code = call_claude(strategy_src, program_src, history)
    except Exception as e:
        print(f"[loop] claude error: {e}")
        return "claude_error", best_metric

    print(f"[loop] proposed: {description}")
    STRATEGY.write_text(new_code, encoding="utf-8")
    sha = git_commit_strategy(description)
    if sha is None:
        # Claude regenerated bytes-identical strategy.py — degenerate mutation.
        # No commit was made; nothing to revert. Skip the backtest entirely.
        print("[loop] no-change → skipping (Claude returned identical strategy)")
        return "noop", best_metric
    print(f"[loop] committed {sha}, running backtest…")

    # Set DSR benchmark from prior-trial variance before invoking backtest.py.
    # Backtest reads DSR_BENCHMARK and emits a deflated PSR ("dsr").
    dsr_benchmark = compute_dsr_benchmark()
    os.environ["DSR_BENCHMARK"] = f"{dsr_benchmark:.10f}"
    if dsr_benchmark > 0:
        print(f"[loop] DSR benchmark (per-bar SR*) = {dsr_benchmark:.6f}")

    metrics = run_backtest()
    sharpe = metrics["val_sharpe"]
    calmar = metrics["calmar"]
    dsr = metrics.get("dsr", 0.0)
    max_dd = metrics["max_drawdown"]
    n_trades = metrics["total_trades"]
    win_rate = metrics["win_rate"]

    score = float(metrics.get(OPTIMIZE_METRIC, 0.0))

    print(
        f"[loop] result: sharpe={sharpe:.4f}  calmar={calmar:.4f}  dsr={dsr:.4f}  "
        f"max_dd={max_dd:.2f}%  trades={n_trades}  win_rate={win_rate:.3f}  "
        f"→ {OPTIMIZE_METRIC}={score:.4f}"
    )

    if sharpe == 0.0 and n_trades == 0:
        status = "crash"
    elif max_dd >= MAX_DRAWDOWN_LIMIT:
        status = "discard"
    elif n_trades < MIN_TRADES:
        status = "discard"
    elif DSR_GATE_THRESHOLD > 0 and dsr < DSR_GATE_THRESHOLD:
        # Multiple-testing gate: reject "lucky" candidates that pass raw Sharpe
        # but fail the trial-count-adjusted significance check.
        status = "discard"
        print(f"[loop] DSR gate: {dsr:.4f} < {DSR_GATE_THRESHOLD:.4f} → reject")
    elif score > best_metric + KEEP_THRESHOLD:
        status = "keep"
    else:
        status = "discard"

    if status != "keep":
        git_reset_last()
        print(f"[loop] {status} → reverted")
    else:
        best_metric = score
        print(f"[loop] KEEP — new best {OPTIMIZE_METRIC}={score:.4f}")

    append_result(sha, metrics, status, description)
    return status, best_metric


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "ERROR: OPENROUTER_API_KEY not set. Get a key at https://openrouter.ai/keys "
            "and add it to .env (see .env.example).",
            file=sys.stderr,
        )
        return 2

    # Layer 1 — checkpoint anything left dirty by a prior aborted session
    # so the dirty-check below sees a clean tree.
    git_commit_results()

    if git_dirty():
        print("ERROR: git working tree is dirty. Commit or stash first.", file=sys.stderr)
        return 2

    _ensure_results()
    best = best_metric_so_far(OPTIMIZE_METRIC)
    gate_msg = f"  DSR_GATE={DSR_GATE_THRESHOLD:.2f}" if DSR_GATE_THRESHOLD > 0 else ""
    print(
        f"[loop] campaign={STRATEGY_REL}  symbols={_bt_module.DEFAULT_SYMBOLS}\n"
        f"[loop] OPTIMIZE_METRIC={OPTIMIZE_METRIC}  starting best={best:.4f}{gate_msg}"
    )
    consecutive_regressions = 0

    # Layer 2 — try/finally guarantees we commit accumulated tsv rows even if
    # the user Ctrl-C's mid-loop or an iteration crashes uncaught. The CI
    # workflow's separate commit step is now redundant.
    rc = 0
    try:
        for i in range(1, args.iters + 1):
            print(f"\n========== iteration {i}/{args.iters} ==========")
            status, best = one_iteration(best)

            if status == "keep":
                consecutive_regressions = 0
            elif status == "claude_error":
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

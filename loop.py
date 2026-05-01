"""Autoresearch orchestrator — the karpathy-style loop.

Each iteration:
  1. Read strategy.py + last 10 rows of results/results.tsv + program.md
  2. Ask Claude for a single mutation (returns description + new strategy.py)
  3. Write strategy.py, git commit
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

Requires:
    ANTHROPIC_API_KEY in env (or .env file)
    A clean git working tree on entry (so we can reset cleanly)

Environment knobs (all optional):
    OPTIMIZE_METRIC        val_sharpe (default) | calmar | dsr
    DSR_GATE_THRESHOLD     reject if dsr below this; 0 disables (default)
    SYMBOLS                comma-sep parquet stems under data/ (e.g.
                           "crypto/BTC_USDT_4h" or "stocks/TSLA_1d,stocks/NVDA_1d").
                           N=1 → single mode; N≥2 → basket mode with overfit penalty.
    CLAUDE_MODEL           override the default Haiku model
    OPENAI_*  / similar    not used; this loop only calls Anthropic

Usage:
    python loop.py --iters 50
    python loop.py --iters 1   # single experiment, useful for debugging
"""
from __future__ import annotations

import argparse
import json
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
STRATEGY = ROOT / "strategy.py"
PROGRAM = ROOT / "program.md"
RESULTS_DIR = ROOT / "results"
RUN_LOG = RESULTS_DIR / "run.log"
LAST_CRASH = RESULTS_DIR / "last_crash.json"
RESULTS_DIR.mkdir(exist_ok=True)

# RESULTS path is per-campaign — derived from active SYMBOLS × val-window so
# each (asset, window) combo gets its own history. Switching SYMBOLS env var
# does not clobber prior research; both files coexist under results/.
import backtest as _bt_module
RESULTS = _bt_module.results_path()

PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PYTHON).exists():
    PYTHON = sys.executable

DEFAULT_MODEL = "claude-haiku-4-5"
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


def git_short_sha() -> str:
    return _git("rev-parse", "--short=7", "HEAD")


def git_commit_strategy(description: str) -> str | None:
    """Commit strategy.py. Returns None if Claude wrote bytes-identical content
    (no diff) — the caller must treat this as a no-op iteration, not a crash."""
    _git("add", str(STRATEGY))
    diff = _git("diff", "--cached", "--name-only")
    if not diff.strip():
        return None
    _git("commit", "-m", f"experiment: {description}", "--no-verify")
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
# only `status=crash` in the history row — no traceback, no idea what
# broke. We persist the crashed source + stderr tail to last_crash.json
# so the *next* call_claude can show Claude exactly what failed.
#
# Lifecycle:
#   crash   → write last_crash.json (overwrites prior crash if any)
#   keep    → clear (we moved past the bug)
#   discard → clear (LLM moved on; old crash no longer relevant)
#   noop    → leave as-is (no real attempt was made)


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


def save_last_crash(src: str, stderr: str, description: str, sha: str | None) -> None:
    LAST_CRASH.write_text(
        json.dumps({
            "src": src,
            "stderr": stderr[-2000:],  # tail; tracebacks are at the end
            "description": description,
            "commit": sha or "",
        }),
        encoding="utf-8",
    )


def load_last_crash() -> dict | None:
    if not LAST_CRASH.exists():
        return None
    try:
        return json.loads(LAST_CRASH.read_text(encoding="utf-8"))
    except Exception:
        return None


def clear_last_crash() -> None:
    if LAST_CRASH.exists():
        try:
            LAST_CRASH.unlink()
        except Exception:
            pass


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


# ──────────────────────────── Claude call ──────────────────────────────

DESCRIPTION_RE = re.compile(r"##\s*Description\s*\n+(.+?)(?=\n##|\Z)", re.DOTALL)
CODE_RE = re.compile(r"##\s*strategy\.py\s*\n+```python\s*\n(.*?)\n```", re.DOTALL)


def call_claude(
    strategy_src: str,
    program_src: str,
    history: list[dict],
    last_crash: dict | None = None,
) -> tuple[str, str]:
    """Returns (description, new_strategy_source).

    If last_crash is provided AND the most recent history row is a crash, the
    crashed strategy.py source + traceback are included in the prompt so
    Claude can fix the actual bug instead of guessing from `status=crash`."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

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
        f"# Current strategy.py\n```python\n{strategy_src}\n```\n\n"
        f"# Last {len(history)} experiments (results.tsv tail)\n```\n{history_block}\n```\n\n"
    )

    # If the immediately-previous iteration crashed, show Claude the exact code
    # that broke and the traceback. Without this, the loop wastes iterations
    # re-proposing the same broken pattern.
    if last_crash and history and history[-1].get("status") == "crash":
        user_msg += (
            "# LAST ITERATION CRASHED — your previous mutation broke. "
            "It has already been reverted. Diagnose and fix the bug; "
            "do not repropose the same code.\n\n"
            f"## Crashed strategy.py (commit {last_crash.get('commit', '?')})\n"
            f"```python\n{last_crash['src']}\n```\n\n"
            f"## Traceback / stderr (tail)\n```\n{last_crash['stderr']}\n```\n\n"
        )

    user_msg += "Propose ONE change. Reply with the two required sections only."

    model = os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=program_src,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))

    desc_m = DESCRIPTION_RE.search(text)
    code_m = CODE_RE.search(text)
    if not desc_m or not code_m:
        raise ValueError(f"could not parse Claude response:\n{text[:1000]}")

    description = desc_m.group(1).strip().splitlines()[0][:200]
    new_code = code_m.group(1)
    return description, new_code


# ──────────────────────────────── main ─────────────────────────────────

def one_iteration(best_metric: float) -> tuple[str, float]:
    """Run one experiment. Returns (status, new_best_for_active_metric)."""
    strategy_src = STRATEGY.read_text(encoding="utf-8")
    program_src = PROGRAM.read_text(encoding="utf-8")
    history = last_n_rows(10)
    last_crash = load_last_crash()

    if last_crash and history and history[-1].get("status") == "crash":
        print(f"[loop] feeding back last crash (commit {last_crash.get('commit', '?')}) to Claude")

    print(f"\n[loop] optimize={OPTIMIZE_METRIC}  best_so_far={best_metric:.4f}  asking Claude…")
    try:
        description, new_code = call_claude(strategy_src, program_src, history, last_crash=last_crash)
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

    # Persist crash artifacts BEFORE git reset so the next iteration can show
    # Claude exactly what failed. On keep/discard, clear stale crash blob.
    if status == "crash":
        save_last_crash(new_code, _read_stderr_tail(), description, sha)
    else:
        clear_last_crash()

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

    if not os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set (env or .env file)", file=sys.stderr)
        return 2

    if git_dirty():
        print("ERROR: git working tree is dirty. Commit or stash first.", file=sys.stderr)
        return 2

    _ensure_results()
    best = best_metric_so_far(OPTIMIZE_METRIC)
    gate_msg = f"  DSR_GATE={DSR_GATE_THRESHOLD:.2f}" if DSR_GATE_THRESHOLD > 0 else ""
    print(f"[loop] OPTIMIZE_METRIC={OPTIMIZE_METRIC}  starting best={best:.4f}{gate_msg}")
    consecutive_regressions = 0

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
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

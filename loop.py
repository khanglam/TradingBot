"""Autoresearch orchestrator — the karpathy-style loop.

Each iteration:
  1. Read strategy.py + last 10 rows of results.tsv + program.md
  2. Ask Claude for a single mutation (returns description + new strategy.py)
  3. Write strategy.py, git commit
  4. Run backtest.py, parse summary block
  5. If val_sharpe improves AND constraints pass → keep
     Else                                       → git reset --hard HEAD~1
  6. Append a row to results.tsv
  7. Repeat until --iters reached or 3 consecutive regressions

Requires:
    ANTHROPIC_API_KEY in env (or .env file)
    A clean git working tree on entry (so we can reset cleanly)

Usage:
    python loop.py --iters 50
    python loop.py --iters 1   # single experiment, useful for debugging
"""
from __future__ import annotations

import argparse
import os
import re
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
RESULTS = ROOT / "results.tsv"
RUN_LOG = ROOT / "run.log"

PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PYTHON).exists():
    PYTHON = sys.executable

DEFAULT_MODEL = "claude-haiku-4-5"
MAX_OUTPUT_TOKENS = 8000

KEEP_THRESHOLD = 0.0  # require strictly > best so far
MAX_DRAWDOWN_LIMIT = 30.0
MIN_TRADES = 20
MAX_REGRESSIONS = 3

# OPTIMIZE_METRIC env var picks the keep/discard scalar.
# Allowed: "val_sharpe" (default; backtesting.py reported Sharpe)
#         "calmar"     (return / max_drawdown — return-aware)
# Both are logged to results.tsv either way; this just chooses the gradient
# the loop hill-climbs against.
ALLOWED_METRICS = {"val_sharpe", "calmar"}
OPTIMIZE_METRIC = os.environ.get("OPTIMIZE_METRIC", "val_sharpe")
if OPTIMIZE_METRIC not in ALLOWED_METRICS:
    raise SystemExit(
        f"OPTIMIZE_METRIC must be one of {sorted(ALLOWED_METRICS)}, got {OPTIMIZE_METRIC!r}"
    )


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


def git_commit_strategy(description: str) -> str:
    _git("add", str(STRATEGY))
    _git("commit", "-m", f"experiment: {description}", "--no-verify")
    return git_short_sha()


def git_reset_last() -> None:
    _git("reset", "--hard", "HEAD~1")


# ──────────────────────────── results.tsv ──────────────────────────────

# Schema columns in fixed order. Append-only — never reorder. Adding a new
# column? Append it to the end and migrate existing files.
RESULTS_COLS = [
    "commit", "val_sharpe", "sortino", "sharpe_ann_4h", "calmar", "psr",
    "skew", "kurtosis", "max_drawdown", "win_rate", "total_trades",
    "status", "description",
]
RESULTS_HEADER = "\t".join(RESULTS_COLS) + "\n"
LEGACY_COLS = [
    "commit", "val_sharpe", "max_drawdown", "win_rate", "total_trades",
    "status", "description",
]


def _ensure_results() -> None:
    """Create results.tsv with new schema if missing, or migrate legacy files
    in place by padding old rows with empty strings for the new columns."""
    if not RESULTS.exists():
        RESULTS.write_text(RESULTS_HEADER, encoding="utf-8")
        return

    lines = RESULTS.read_text(encoding="utf-8").splitlines()
    if not lines:
        RESULTS.write_text(RESULTS_HEADER, encoding="utf-8")
        return

    header_cols = lines[0].split("\t")
    if header_cols == RESULTS_COLS:
        return  # already on new schema

    if header_cols == LEGACY_COLS:
        # Legacy → new: pad with empty values for the inserted columns.
        # Old order:  commit, val_sharpe, max_drawdown, win_rate, total_trades, status, description
        # New order:  commit, val_sharpe, sortino, sharpe_ann_4h, calmar, psr, skew, kurtosis,
        #             max_drawdown, win_rate, total_trades, status, description
        backup = RESULTS.with_suffix(".tsv.legacy")
        if not backup.exists():
            backup.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"[loop] migrated {RESULTS.name} (legacy backup at {backup.name})")
        new_lines = [RESULTS_HEADER.rstrip("\n")]
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            commit, val_sharpe, max_dd, win_rate, n_trades, status, desc = parts[:7]
            # insert empty placeholders for sortino, sharpe_ann_4h, calmar, psr, skew, kurtosis
            new_row = [commit, val_sharpe, "", "", "", "", "", "",
                       max_dd, win_rate, n_trades, status, desc]
            new_lines.append("\t".join(new_row))
        RESULTS.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return

    raise RuntimeError(
        f"results.tsv has unrecognized header: {header_cols}. "
        f"Expected legacy or new schema. Move/rename it manually."
    )


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
    "calmar": float, "psr": float, "skew": float, "kurtosis": float,
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
    """Run backtest.py in a subprocess, parse summary block, return metrics."""
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


# ──────────────────────────── Claude call ──────────────────────────────

DESCRIPTION_RE = re.compile(r"##\s*Description\s*\n+(.+?)(?=\n##|\Z)", re.DOTALL)
CODE_RE = re.compile(r"##\s*strategy\.py\s*\n+```python\s*\n(.*?)\n```", re.DOTALL)


def call_claude(strategy_src: str, program_src: str, history: list[dict]) -> tuple[str, str]:
    """Returns (description, new_strategy_source)."""
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
        f"Propose ONE change. Reply with the two required sections only."
    )

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

    print(f"\n[loop] optimize={OPTIMIZE_METRIC}  best_so_far={best_metric:.4f}  asking Claude…")
    try:
        description, new_code = call_claude(strategy_src, program_src, history)
    except Exception as e:
        print(f"[loop] claude error: {e}")
        return "claude_error", best_metric

    print(f"[loop] proposed: {description}")
    STRATEGY.write_text(new_code, encoding="utf-8")
    sha = git_commit_strategy(description)
    print(f"[loop] committed {sha}, running backtest…")

    metrics = run_backtest()
    sharpe = metrics["val_sharpe"]
    calmar = metrics["calmar"]
    max_dd = metrics["max_drawdown"]
    n_trades = metrics["total_trades"]
    win_rate = metrics["win_rate"]

    score = float(metrics.get(OPTIMIZE_METRIC, 0.0))

    print(
        f"[loop] result: sharpe={sharpe:.4f}  calmar={calmar:.4f}  "
        f"max_dd={max_dd:.2f}%  trades={n_trades}  win_rate={win_rate:.3f}  "
        f"→ {OPTIMIZE_METRIC}={score:.4f}"
    )

    if sharpe == 0.0 and n_trades == 0:
        status = "crash"
    elif max_dd >= MAX_DRAWDOWN_LIMIT:
        status = "discard"
    elif n_trades < MIN_TRADES:
        status = "discard"
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
    print(f"[loop] OPTIMIZE_METRIC={OPTIMIZE_METRIC}  starting best={best:.4f}")
    consecutive_regressions = 0

    for i in range(1, args.iters + 1):
        print(f"\n========== iteration {i}/{args.iters} ==========")
        status, best = one_iteration(best)

        if status == "keep":
            consecutive_regressions = 0
        elif status == "claude_error":
            time.sleep(10)
        else:
            consecutive_regressions += 1
            if consecutive_regressions >= MAX_REGRESSIONS:
                print(f"\n[loop] {MAX_REGRESSIONS} consecutive regressions — freezing for review.")
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

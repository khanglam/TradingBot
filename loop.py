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

RESULTS_HEADER = "commit\tval_sharpe\tmax_drawdown\twin_rate\ttotal_trades\tstatus\tdescription\n"


def _ensure_results() -> None:
    if not RESULTS.exists():
        RESULTS.write_text(RESULTS_HEADER, encoding="utf-8")


def append_result(
    sha: str,
    val_sharpe: float,
    max_dd: float,
    win_rate: float,
    n_trades: int,
    status: str,
    description: str,
) -> None:
    _ensure_results()
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(
            f"{sha}\t{val_sharpe:.6f}\t{max_dd:.2f}\t"
            f"{win_rate:.3f}\t{n_trades}\t{status}\t{description}\n"
        )


def best_val_sharpe_so_far() -> float:
    if not RESULTS.exists():
        return 0.0
    best = 0.0
    for line in RESULTS.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 6 and parts[5] == "keep":
            try:
                best = max(best, float(parts[1]))
            except ValueError:
                continue
    return best


def last_n_rows(n: int = 10) -> list[dict]:
    if not RESULTS.exists():
        return []
    lines = RESULTS.read_text(encoding="utf-8").splitlines()[1:]
    out = []
    for line in lines[-n:]:
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        out.append({
            "commit": parts[0],
            "val_sharpe": parts[1],
            "max_drawdown": parts[2],
            "win_rate": parts[3],
            "total_trades": parts[4],
            "status": parts[5],
            "description": parts[6],
        })
    return out


# ─────────────────────────── backtest runner ───────────────────────────

SUMMARY_RE = re.compile(
    r"^val_sharpe:\s+([-\d.]+)\s*$.*?"
    r"^max_drawdown:\s+([-\d.]+)\s*$.*?"
    r"^win_rate:\s+([-\d.]+)\s*$.*?"
    r"^total_trades:\s+(\d+)\s*$",
    re.MULTILINE | re.DOTALL,
)


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

    m = SUMMARY_RE.search(out)
    if not m:
        return {"val_sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "total_trades": 0}
    return {
        "val_sharpe": float(m.group(1)),
        "max_drawdown": float(m.group(2)),
        "win_rate": float(m.group(3)),
        "total_trades": int(m.group(4)),
    }


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

def one_iteration(best_sharpe: float) -> tuple[str, float]:
    """Run one experiment. Returns (status, new_best_sharpe)."""
    strategy_src = STRATEGY.read_text(encoding="utf-8")
    program_src = PROGRAM.read_text(encoding="utf-8")
    history = last_n_rows(10)

    print(f"\n[loop] best_so_far={best_sharpe:.4f}  asking Claude…")
    try:
        description, new_code = call_claude(strategy_src, program_src, history)
    except Exception as e:
        print(f"[loop] claude error: {e}")
        return "claude_error", best_sharpe

    print(f"[loop] proposed: {description}")
    STRATEGY.write_text(new_code, encoding="utf-8")
    sha = git_commit_strategy(description)
    print(f"[loop] committed {sha}, running backtest…")

    metrics = run_backtest()
    sharpe = metrics["val_sharpe"]
    max_dd = metrics["max_drawdown"]
    n_trades = metrics["total_trades"]
    win_rate = metrics["win_rate"]

    print(
        f"[loop] result: sharpe={sharpe:.4f}  max_dd={max_dd:.2f}%  "
        f"trades={n_trades}  win_rate={win_rate:.3f}"
    )

    if sharpe == 0.0 and n_trades == 0:
        status = "crash"
    elif max_dd >= MAX_DRAWDOWN_LIMIT:
        status = "discard"
    elif n_trades < MIN_TRADES:
        status = "discard"
    elif sharpe > best_sharpe + KEEP_THRESHOLD:
        status = "keep"
    else:
        status = "discard"

    if status != "keep":
        git_reset_last()
        print(f"[loop] {status} → reverted")
    else:
        best_sharpe = sharpe
        print(f"[loop] KEEP — new best {sharpe:.4f}")

    append_result(sha, sharpe, max_dd, win_rate, n_trades, status, description)
    return status, best_sharpe


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
    best = best_val_sharpe_so_far()
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

"""Promotion gate — copy a campaign's candidate strategy onto main if it
clears the validation gauntlet.

The autoresearch loop runs on autoresearch/<campaign> branches and never
touches main. main's strategies/<campaign>.py is the frozen version that
scan.py / live_trade.py read. This script is the only path between the two:

  1. Read the candidate strategy from autoresearch/<campaign> HEAD.
  2. If identical to frozen → no-op.
  3. Run backtest on candidate (val window) — must beat frozen by PROMOTION_MARGIN.
  4. Run backtest on candidate (lockbox window) — sanity floors:
       sharpe ≥ LOCKBOX_MIN_SHARPE  and  max_drawdown ≤ LOCKBOX_MAX_DD.
  5. If all pass → overwrite the frozen file in the main working tree.
     The caller (promote.yml) commits + pushes.

Exit codes:
  0 → promoted (caller should commit + push)
  1 → no-op (no candidate, no improvement, or sanity-floor failure)
  2 → error (unknown campaign, missing branch, etc.)

Usage:
    python promote.py --campaign stocks
    python promote.py --campaign crypto

The script must be run from a checkout of main with autoresearch/<campaign>
fetched (the workflow handles the fetch).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

CAMPAIGN_FILES = {
    "stocks": "strategies/stocks.py",
    "crypto": "strategies/crypto.py",
}

# Candidate val-window OPTIMIZE_METRIC must exceed frozen's by this margin
# (in absolute metric units). Conservative — promotions should be rare.
PROMOTION_MARGIN = float(os.environ.get("PROMOTION_MARGIN", "0.05"))

# Lockbox sanity floors. The candidate doesn't have to win on lockbox; it
# just has to show it isn't catastrophically broken there.
LOCKBOX_MIN_SHARPE = float(os.environ.get("LOCKBOX_MIN_SHARPE", "0.0"))
LOCKBOX_MAX_DD = float(os.environ.get("LOCKBOX_MAX_DD", "50.0"))


def _git(*args: str) -> str:
    res = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                         text=True, check=True)
    return res.stdout.strip()


_NUMERIC_KEYS = {"val_sharpe", "max_drawdown", "calmar", "total_trades",
                 "total_return_pct", "dsr", "sharpe_ann_4h"}
_SUMMARY_RE = re.compile(r"^([a-z_0-9]+):\s+([-\d.]+)\s*$")


def _parse_summary(stdout: str) -> dict[str, float]:
    metrics = {k: 0.0 for k in _NUMERIC_KEYS}
    in_block = False
    for line in stdout.splitlines():
        s = line.strip()
        if s == "---":
            in_block = not in_block
            continue
        if not in_block:
            continue
        m = _SUMMARY_RE.match(s)
        if m and m.group(1) in _NUMERIC_KEYS:
            try:
                metrics[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return metrics


def _run_backtest(strategy_path: Path, window: str, campaign: str) -> dict[str, float]:
    env = os.environ.copy()
    env["STRATEGY_FILE"] = str(strategy_path)
    env["CAMPAIGN"] = campaign
    proc = subprocess.run(
        [sys.executable, "backtest.py", "--window", window],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
    return _parse_summary(proc.stdout)


def promote(campaign: str) -> int:
    if campaign not in CAMPAIGN_FILES:
        print(f"unknown campaign: {campaign}", file=sys.stderr)
        return 2

    rel_path = CAMPAIGN_FILES[campaign]
    frozen_path = ROOT / rel_path
    branch = f"autoresearch/{campaign}"

    try:
        candidate_src = _git("show", f"{branch}:{rel_path}")
        candidate_sha = _git("rev-parse", "--short=7", branch)
    except subprocess.CalledProcessError as e:
        print(f"[promote] {campaign}: cannot read {branch} — fetch it first ({e})",
              file=sys.stderr)
        return 2

    frozen_src = frozen_path.read_text(encoding="utf-8") if frozen_path.exists() else ""
    if candidate_src == frozen_src:
        print(f"[promote] {campaign}: candidate identical to frozen — nothing to do")
        return 1

    tmp_dir = ROOT / "_promote_tmp"
    tmp_dir.mkdir(exist_ok=True)
    candidate_path = tmp_dir / f"{campaign}.py"
    candidate_path.write_text(candidate_src, encoding="utf-8")

    print(f"[promote] {campaign}: candidate {candidate_sha} — running val backtest…")
    cand_val = _run_backtest(candidate_path, "val", campaign)
    print(f"[promote] {campaign}: candidate val sharpe={cand_val['val_sharpe']:.4f} "
          f"dd={cand_val['max_drawdown']:.2f}% trades={int(cand_val['total_trades'])}")

    if frozen_path.exists():
        print(f"[promote] {campaign}: frozen — running val backtest…")
        frozen_val = _run_backtest(frozen_path, "val", campaign)
    else:
        frozen_val = {"val_sharpe": 0.0}
    print(f"[promote] {campaign}: frozen   val sharpe={frozen_val['val_sharpe']:.4f}")

    if cand_val["val_sharpe"] <= frozen_val["val_sharpe"] + PROMOTION_MARGIN:
        print(f"[promote] {campaign}: candidate does not beat frozen by "
              f"{PROMOTION_MARGIN:.4f} — skipping")
        return 1

    print(f"[promote] {campaign}: candidate clears val gate — running lockbox backtest…")
    cand_lock = _run_backtest(candidate_path, "lockbox", campaign)
    print(f"[promote] {campaign}: candidate lockbox sharpe={cand_lock['val_sharpe']:.4f} "
          f"dd={cand_lock['max_drawdown']:.2f}% trades={int(cand_lock['total_trades'])}")

    if cand_lock["val_sharpe"] < LOCKBOX_MIN_SHARPE:
        print(f"[promote] {campaign}: lockbox sharpe < {LOCKBOX_MIN_SHARPE} — refusing")
        return 1
    if cand_lock["max_drawdown"] > LOCKBOX_MAX_DD:
        print(f"[promote] {campaign}: lockbox max_dd > {LOCKBOX_MAX_DD} — refusing")
        return 1

    frozen_path.write_text(candidate_src, encoding="utf-8")
    print(f"[promote] {campaign}: PROMOTED {rel_path} from candidate {candidate_sha}")
    print(f"[promote] {campaign}: suggested commit message: 'promote {campaign} {candidate_sha}'")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True, choices=list(CAMPAIGN_FILES))
    args = p.parse_args()
    return promote(args.campaign)


if __name__ == "__main__":
    sys.exit(main())

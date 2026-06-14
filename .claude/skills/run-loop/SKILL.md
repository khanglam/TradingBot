---
name: run-loop
description: Run TradingBot's local karpathy-style autoresearch loop (loop.py) on dev to mutate strategy files. Use whenever the user wants to loop, autoresearch, optimize a strategy, run N iterations, continue looping, run the research loop locally, or mentions loop.py with stocks or crypto — even if they only say "run 50 loops" or "keep looping". Defaults to crypto campaign and 50 iterations. Do NOT use for harness perf tuning (use optimize-loop) or environment setup (use init-local-dev).
---

# /run-loop [campaign] [N]

Run the **local autoresearch loop** (`loop.py`) on the **`dev`** branch. Each iteration asks the LLM (via OpenRouter) for one mutation to the active strategy file, backtests train+val, and keeps or reverts the commit.

This is the same harness GitHub Actions runs nightly — use it when the user wants their **local agent** to drive strategy research, not when they want a Cursor-only manual mutation session (that is a different workflow).

## Parse arguments

The full argument string is: `$ARGUMENTS`

Extract **campaign** and **N**:

| Input | Campaign | Iterations |
|-------|----------|------------|
| (empty) | `crypto` | `50` |
| `20` | `crypto` | `20` |
| `crypto` | `crypto` | `50` |
| `stocks` | `stocks` | `50` |
| `crypto 100` | `crypto` | `100` |
| `stocks 25` | `stocks` | `25` |

Rules:
- If the first token is `crypto` or `stocks`, that is the campaign.
- If a token is a positive integer, that is **N**.
- Campaign must be exactly `crypto` or `stocks` (maps to `[crypto]` / `[stocks]` in `configs.toml`).
- Default campaign: **`crypto`**. Default **N**: **`50`** (matches `loop.py` default).

Print once before starting: `[run-loop] campaign=<c>  iters=<N>  strategy=strategies/<c>.py`

## Phase 0 — Pre-flight (stop on failure)

Run from **project root** (directory containing `loop.py`, `backtest.py`, `.git`).

### 0a — Environment

1. **uv + venv** — `uv` on PATH and `.venv/` present. If missing → tell user to run `/init-local-dev`.
2. **`.env`** — `OPENROUTER_API_KEY` must be set (loop refuses without it). Point to `.env.example` if missing.
3. **Data** — at least one `.parquet` under `data/`. If missing → `uv run python data_fetch.py` with `CAMPAIGN` set.
4. **Branch** — must be `dev`. If not → `git checkout dev` (or stop and ask user).

### 0b — Working tree

Run `git status --porcelain`.

`loop.py` **auto-commits pending `results/*.tsv` rows** at session start (`git_commit_results`). Do **not** stash the experiment ledger to satisfy the clean-tree check — that hides history.

If the tree is still dirty after understanding what's modified:
- **`results/*.tsv` only** → proceed; `loop.py` will checkpoint those rows on start.
- **Other paths** (e.g. uncommitted `strategies/*.py`, manual edits) → stop and ask the user to commit or stash **those** files. Never stash results TSV to "clean up."

### 0c — Collision warning

If local time is near **03:00 America/Los_Angeles**, warn that GHA may push to `dev` concurrently. User can pause the remote workflow or run anyway.

## Phase 1 — Run the loop

Set env for the whole session (use `uv run python` — works on all platforms):

**Windows (PowerShell):**
```powershell
$env:CAMPAIGN = "<campaign>"
$log = "results/loop-<campaign>-$(Get-Date -Format 'yyyyMMdd-HHmmss').log"
uv run python loop.py --iters <N> 2>&1 | Tee-Object -FilePath $log
```

**macOS / Linux:**
```bash
export CAMPAIGN="<campaign>"
log="results/loop-<campaign>-$(date +%Y%m%d-%H%M%S).log"
uv run python loop.py --iters <N> 2>&1 | tee "$log"
```

### Long runs

For **N > 10**, run in the **background** so the user can keep working. Tell them:
- Log path under `results/loop-*.log`
- How to watch: `Get-Content results/loop-....log -Wait` (PowerShell) or `tail -f` (Unix)
- How to stop: run `scripts/stop_loops.py` from this skill directory (see below)

### What loop.py does (so you can report status)

- Mutates `strategies/crypto.py` or `strategies/stocks.py` per `CAMPAIGN`
- Appends rows to the campaign's `results/<symbols>_<val-window>.tsv`
- **Keep** → strategy commit stays; **discard/crash** → `git reset --hard HEAD~1`
- Commits accumulated TSV rows in a final chore commit on exit (including Ctrl+C via `finally`)

## Phase 2 — After the run

1. Read the log tail for `[loop] KEEP`, discard reasons, and final exit code.
2. Report: iterations attempted, keeps (grep `KEEP —` in log), current `git log --oneline -3`, path to results TSV.
3. Remind: **promotion is separate** — train/val wins do not paper-trade until `promote.py` PASS on the commit they deploy.

Do **not** push to remote unless the user asks.

## Stop loops

If the user says stop / cancel / abort loops, run from project root:

```bash
python .claude/skills/run-loop/scripts/stop_loops.py
```

Then confirm no `loop.py` or `backtest.py` processes remain for this repo.

## Related skills

| Skill | When |
|-------|------|
| `/init-local-dev` | No venv, wrong branch, missing deps |
| `/optimize-loop` | Speed up `loop.py` / backtest harness — never mutates strategies |
| `/sync-branches` | Merge `dev` → `main` after research |

## Common mistakes (learned from production)

1. **Stashing `results/*.tsv`** before loop — wrong; loop checkpoints TSV itself; stashing drops the ledger from disk.
2. **Running on `main`** — loop refuses; switch to `dev`.
3. **Expecting instant keeps** — ~10% keep rate is normal; val gate blocks most train gains.
4. **Confusing with Cursor manual autoresearch** — that path edits strategy in-agent without OpenRouter; this skill runs **`loop.py`**.

# Migration: branch-per-campaign + promotion gate

One-time setup to move from "loop commits to main" to the new model. Run
these commands yourself — automation deliberately does not push branches
or rewrite main on your behalf.

## 1. Push the new infrastructure on main

The current branch (`main`) now contains the new `loop.yml`, `promote.yml`,
`promote.py`, and the loop-on-main guard. Commit them as one change:

```bash
git status                         # confirm what's pending
git add .github/workflows/loop.yml \
        .github/workflows/promote.yml \
        promote.py loop.py \
        CLAUDE.md MIGRATION.md .gitignore
git rm karpathy-auto-research      # the broken submodule gitlink is removed
git commit -m "infra: branch-per-campaign loop + promotion gate"
git push origin main
```

## 2. Create the campaign branches from the current main

Each campaign branch is a single-writer space for the loop. Forking from
the current main means each branch starts with the latest frozen strategy
(its own starting candidate) plus its existing results history.

```bash
git checkout -b autoresearch/stocks main
git push -u origin autoresearch/stocks

git checkout -b autoresearch/crypto main
git push -u origin autoresearch/crypto

git checkout main
```

## 3. (Optional) Strip old result tsvs from main

Going forward, results tsvs live on the campaign branches. Main only needs
the frozen strategy files. Remove the tsvs from main to avoid drift:

```bash
git checkout main
git rm results/crypto-BTC_USDT_4h_2022-2024.tsv \
       results/stocks-NVDA-PYPL-TSLA_1d_2020-2024.tsv
git commit -m "chore: results tsvs now live on campaign branches only"
git push origin main
```

If you skip this, the tsvs on main will simply go stale; no functional
problem, just clutter.

## 4. Verify CI

- Trigger `.github/workflows/loop.yml` manually with `campaign: stocks`,
  `iters: 1`. Confirm the run checks out `autoresearch/stocks`, makes one
  experiment commit, and pushes only to that branch.
- After the loop has produced at least one keep, trigger
  `.github/workflows/promote.yml` manually with `campaign: stocks`.
  Confirm it either promotes (commit on main) or logs "candidate does
  not beat frozen" and exits cleanly.

## 5. Local workflow

```bash
# Research a campaign:
git checkout autoresearch/stocks
CAMPAIGN=stocks python loop.py --iters 5

# Manually validate + promote:
git checkout main
git fetch origin "autoresearch/stocks:autoresearch/stocks"
python promote.py --campaign stocks
# If exit 0: review the diff, commit the strategies/stocks.py change yourself.
```

The loop refuses to run on `main` (set `ALLOW_LOOP_ON_MAIN=1` to override
in the rare case you want to commit directly there).

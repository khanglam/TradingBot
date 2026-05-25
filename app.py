"""TradingBot dashboard.

Run:
    .venv/Scripts/python.exe app.py
Then open http://127.0.0.1:8000

Two files, no build step:
    app.py        ← this file (FastAPI server)
    index.html    ← entire UI inline (HTML + Tailwind + Alpine + Chart.js + CSS)
"""
from __future__ import annotations

import asyncio
from datetime import datetime
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Load .env so OPENROUTER_MODEL / OPENROUTER_API_KEY are visible to the dashboard.
# Every other script in the project does this; app.py was the odd one out, which
# is why the model badge in the UI showed the hardcoded default instead of
# whatever the user set in .env.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "web" / "index.html"
WORKTREE_ROOT = ROOT / ".worktrees"

_active_campaign: str = "crypto"
os.environ.setdefault("CAMPAIGN", "crypto")  # backtest.py reads this at import time


def _results_path() -> Path:
    """Per-campaign results.tsv path inside the active campaign's worktree.
    Recomputed each call so the dashboard follows env changes without restart."""
    import backtest as bt
    rel = bt.results_path().relative_to(ROOT)
    return _campaign_root() / rel
PROGRAM = ROOT / "program.md"
DATA_DIR = ROOT / "data"


def _strategy_path() -> Path:
    """Per-campaign strategy file path inside the active campaign's worktree."""
    import backtest as bt
    return _campaign_root() / bt.STRATEGY_FILE

PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

app = FastAPI(title="TradingBot Autoresearch UI")


# ────────────────────── campaign worktree helpers ─────────────────────────
# The autoresearch loop runs on autoresearch/<campaign> branches, each
# checked out as its own git worktree under .worktrees/. The dashboard
# never runs `git checkout` on the main repo — it just resolves paths
# into the right worktree based on the active campaign.

def _worktree_for(campaign: str) -> Path:
    return WORKTREE_ROOT / campaign


def _campaign_root(campaign: str | None = None) -> Path:
    """Filesystem root for reads/writes that follow the active campaign.
    Defaults to the campaign worktree; falls back to ROOT if the worktree
    is missing so the dashboard never 500s on a fresh clone — the user
    sees stale main data plus the setup hint instead."""
    if campaign is None:
        campaign = _active_campaign
    wt = _worktree_for(campaign)
    return wt if wt.exists() else ROOT


def _campaign_or_400() -> str:
    if not _active_campaign:
        raise HTTPException(400, "No campaign selected. POST /api/campaign/<name> first.")
    return _active_campaign




# ───────────────────────────── helpers ───────────────────────────────

_NEW_SCHEMA_COLS = [
    "commit", "val_sharpe", "sortino", "sharpe_ann_4h", "calmar", "psr", "dsr",
    "skew", "kurtosis", "max_drawdown", "win_rate", "total_trades",
    "status", "timestamp", "description",
]

# commit sha -> formatted time (dashboard only); avoids repeated git lookups.
_COMMIT_TS_CACHE: dict[str, str | None] = {}


def _format_ts_no_seconds(s: str) -> str:
    """Normalize to YYYY-MM-DD HH:mm (drop seconds / timezone noise)."""
    s = s.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$", s):
        return s
    m = re.match(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})", s)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return s


def _git_commit_time_display(sha: str) -> str | None:
    sha = sha.strip()
    if len(sha) < 7:
        return None
    if sha in _COMMIT_TS_CACHE:
        return _COMMIT_TS_CACHE[sha]
    try:
        r = subprocess.run(
            ["git", "show", "-s", "--format=%cI", sha],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        _COMMIT_TS_CACHE[sha] = None
        return None
    if r.returncode != 0:
        _COMMIT_TS_CACHE[sha] = None
        return None
    raw = r.stdout.strip()
    try:
        disp = datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime(
            "%Y-%m-%d %H:%M"
        )
    except ValueError:
        disp = None
    _COMMIT_TS_CACHE[sha] = disp
    # Rows use 7-char shas — cache resolves full hashes from batched preload.
    if len(sha) > 7:
        _COMMIT_TS_CACHE[sha[:7]] = disp
    return disp


def _warm_git_commit_cache(shas: list[str]) -> None:
    """Fill _COMMIT_TS_CACHE in a few subprocess calls instead of one per row."""
    to_fetch = sorted({s for s in shas if len(s.strip()) >= 7 and s not in _COMMIT_TS_CACHE})
    if not to_fetch:
        return
    step = 50
    for i in range(0, len(to_fetch), step):
        chunk = to_fetch[i : i + step]
        try:
            proc = subprocess.run(
                ["git", "log", "--no-walk=sorted", *chunk, "--pretty=format:%H%n%cI%n"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode != 0:
            continue
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        for j in range(0, len(lines) - 1, 2):
            full_hash, iso_raw = lines[j], lines[j + 1]
            try:
                disp = datetime.fromisoformat(
                    iso_raw.replace("Z", "+00:00")
                ).strftime("%Y-%m-%d %H:%M")
            except ValueError:
                continue
            _COMMIT_TS_CACHE[full_hash] = disp
            if len(full_hash) >= 7:
                _COMMIT_TS_CACHE[full_hash[:7]] = disp


def _row_timestamp(commit: Any, logged: Any) -> str | None:
    if logged is not None and str(logged).strip():
        return _format_ts_no_seconds(str(logged))
    return _git_commit_time_display(str(commit or "").strip()) or None


def _load_results() -> pd.DataFrame:
    RESULTS = _results_path()
    if not RESULTS.exists():
        return pd.DataFrame(columns=_NEW_SCHEMA_COLS)
    df = pd.read_csv(RESULTS, sep="\t")
    numeric_cols = ("val_sharpe", "sortino", "sharpe_ann_4h", "calmar", "psr", "dsr",
                    "skew", "kurtosis", "max_drawdown", "win_rate")
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "total_trades" in df.columns:
        df["total_trades"] = pd.to_numeric(df["total_trades"], errors="coerce").fillna(0).astype(int)
    return df


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def _safe_float(x: Any) -> float | None:
    """NaN/Inf are not valid JSON. Return None so starlette can encode."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _safe_floats(values: Any) -> list[float | None]:
    return [_safe_float(v) for v in values]


def _active_model() -> str:
    """Read OPENROUTER_MODEL env override; else regex DEFAULT_MODEL out of loop.py.
    Avoids importing loop (which has init-time side effects)."""
    if env := os.environ.get("OPENROUTER_MODEL"):
        return env
    try:
        text = (ROOT / "loop.py").read_text(encoding="utf-8")
        m = re.search(r'^\s*DEFAULT_MODEL\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "unknown"


# ─────────────────────── loop process state ──────────────────────────

class LoopProc:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        # Each entry is (monotonic_id, text). IDs only ever increase, so SSE
        # clients can resume from "last id seen" — robust against buffer.clear().
        self.buffer: deque[tuple[int, str]] = deque(maxlen=2000)
        self.next_id: int = 0
        self.lock = threading.Lock()
        self.reader: threading.Thread | None = None
        self.current_iter: int = 0
        self.total_iters: int = 0

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _append(self, line: str) -> None:
        self.buffer.append((self.next_id, line))
        self.next_id += 1

    def start(self, iters: int, campaign: str) -> None:
        with self.lock:
            if self.is_running():
                raise RuntimeError("Loop already running.")
            wt = _worktree_for(campaign)
            if not wt.exists():
                raise RuntimeError(
                    f"Worktree {wt} does not exist. One-time setup: "
                    f"`git worktree add .worktrees/{campaign} autoresearch/{campaign}`"
                )
            self.buffer.clear()
            self.current_iter = 0
            self.total_iters = iters
            self._append(f"[ui] starting loop.py --iters {iters}  (cwd={wt})")
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            env = os.environ.copy()
            env["CAMPAIGN"] = campaign
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            self.proc = subprocess.Popen(
                [str(PYTHON), "loop.py", "--iters", str(int(iters))],
                cwd=wt, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, creationflags=flags,
                encoding="utf-8", errors="replace", env=env,
            )
            self.reader = threading.Thread(target=self._drain, daemon=True)
            self.reader.start()

    def _drain(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        _iter_re = re.compile(r"iteration\s+(\d+)/(\d+)")
        for line in iter(self.proc.stdout.readline, ""):
            line = line.rstrip("\n")
            m = _iter_re.search(line)
            if m:
                self.current_iter = int(m.group(1))
                self.total_iters = int(m.group(2))
            self._append(line)
        rc = self.proc.wait()
        self._append(f"[ui] loop exited with code {rc}")

    def stop(self) -> bool:
        with self.lock:
            if not self.is_running():
                return False
            assert self.proc is not None
            try:
                if sys.platform == "win32":
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.proc.terminate()
            except Exception:
                self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self._append("[ui] loop stopped by user")
            return True


loop_proc = LoopProc()


class PaperTradeProc:
    """Thin wrapper to run live_trade.py as a supervised subprocess.
    Mirrors LoopProc so the same SSE pattern can serve both."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.buffer: deque[tuple[int, str]] = deque(maxlen=2000)
        self.next_id: int = 0
        self.lock = threading.Lock()
        self.reader: threading.Thread | None = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _append(self, line: str) -> None:
        self.buffer.append((self.next_id, line))
        self.next_id += 1

    def start(self, asset: str, symbols: str | None, dry: bool, live: bool) -> None:
        with self.lock:
            if self.is_running():
                raise RuntimeError("Paper trader already running.")
            self.buffer.clear()
            args = [str(PYTHON), "live_trade.py", "--asset", asset]
            if symbols:
                args += ["--symbols", symbols]
            if dry:
                args.append("--dry")
            if live:
                args.append("--live")
            self._append(f"[trade] starting: {' '.join(args[2:])}")
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            self.proc = subprocess.Popen(
                args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, creationflags=flags,
                encoding="utf-8", errors="replace", env=env,
            )
            self.reader = threading.Thread(target=self._drain, daemon=True)
            self.reader.start()

    def _drain(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in iter(self.proc.stdout.readline, ""):
            self._append(line.rstrip("\n"))
        rc = self.proc.wait()
        self._append(f"[trade] exited with code {rc}")

    def stop(self) -> bool:
        with self.lock:
            if not self.is_running():
                return False
            assert self.proc is not None
            try:
                if sys.platform == "win32":
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.proc.terminate()
            except Exception:
                self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self._append("[trade] stopped by user")
            return True


paper_proc = PaperTradeProc()


# ───────────────────────────── routes ────────────────────────────────

@app.get("/")
def index():
    return FileResponse(INDEX_HTML)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/summary")
def summary() -> dict:
    df = _load_results()
    keeps = df[df["status"] == "keep"]
    discards = df[df["status"] == "discard"]
    crashes = df[df["status"] == "crash"]
    best = float(keeps["val_sharpe"].max()) if not keeps.empty else 0.0
    latest = None
    if not df.empty:
        last = df.iloc[-1].to_dict()
        latest = {
            "commit": last["commit"],
            "val_sharpe": float(last["val_sharpe"] or 0.0),
            "max_drawdown": float(last["max_drawdown"] or 0.0),
            "status": last["status"],
            "description": last["description"],
        }
    import backtest as bt_module
    try:
        symbols_spec = _load_campaigns()[_active_campaign]["symbols"]
    except Exception:
        symbols_spec = bt_module.DEFAULT_SYMBOLS
    return {
        "best_sharpe": best,
        "total": int(len(df)),
        "keeps": int(len(keeps)),
        "discards": int(len(discards)),
        "crashes": int(len(crashes)),
        "keep_rate": (len(keeps) / len(df)) if len(df) else 0.0,
        "latest": latest,
        "model": _active_model(),
        "symbols": symbols_spec,
        "val_window": f"{bt_module.VAL_START} → {bt_module.VAL_END}",
    }


@app.get("/api/results")
def results() -> list[dict[str, Any]]:
    rows = _df_to_records(_load_results())
    need_git_ts = sorted(
        {
            str(r["commit"]).strip()
            for r in rows
            if not str(r.get("timestamp") or "").strip()
            and len(str(r.get("commit") or "").strip()) >= 7
        }
    )
    _warm_git_commit_cache(need_git_ts)
    for r in rows:
        r["timestamp"] = _row_timestamp(r.get("commit"), r.get("timestamp"))
    return rows


@app.get("/api/strategy")
def strategy_source() -> dict:
    path = _strategy_path()
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found")
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    return {"path": rel, "source": path.read_text(encoding="utf-8")}


@app.get("/api/program")
def program_source() -> dict:
    if not PROGRAM.exists():
        raise HTTPException(404, "program.md not found")
    return {"source": PROGRAM.read_text(encoding="utf-8")}


@app.get("/api/git-log")
def git_log(n: int = 20) -> dict:
    # Show commit history for the active campaign's strategy file on its
    # research branch. Run `git log` from inside the worktree so HEAD is
    # the campaign branch. Pre-split commits used "strategy.py" — include
    # it as a fallback path so older history still appears.
    import backtest as bt
    rel = bt.STRATEGY_FILE  # e.g. "strategies/stocks.py"
    cwd = _campaign_root()
    try:
        out = subprocess.run(
            ["git", "log", "--pretty=format:%h|%ar|%s", f"-{int(n)}",
             "--", rel, "strategy.py"],
            cwd=cwd, capture_output=True, text=True, check=True,
        ).stdout
        commits = []
        for line in out.splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                commits.append({"hash": parts[0], "when": parts[1], "subject": parts[2]})
        return {"commits": commits}
    except Exception as e:
        return {"commits": [], "error": str(e)}


@app.get("/api/equity")
def get_equity():
    """Run a fast backtest on the best strategy to get the equity curve."""
    # We must run it in a subprocess or a separate thread, but backtesting.py is fast enough
    # for 2-3 assets to just block for 100ms.
    
    # Reload bt_module to pick up any campaign switch or env change.
    # Do NOT load_dotenv(override=True) here — that would clobber env vars
    # set by /api/campaign (campaign switch would reset on every equity call).
    import importlib
    import backtest as bt_module
    importlib.reload(bt_module)
    from backtesting import Backtest
    from backtesting.lib import FractionalBacktest
    # Load the candidate from the active campaign's worktree, not main's
    # frozen copy — this is the dashboard's "current research" view.
    UserStrategy = bt_module.load_strategy_class(_campaign_root() / bt_module.STRATEGY_FILE)

    # Equity curve is single-asset by definition: pick the first symbol
    # the harness would resolve from $SYMBOLS (or DEFAULT_SYMBOLS).
    spec = os.environ.get("SYMBOLS") or bt_module.DEFAULT_SYMBOLS
    resolved = bt_module._resolve_symbols(spec)
    if not resolved:
        raise HTTPException(400, "no symbols resolved from $SYMBOLS")
    import data_fetch
    try:
        data_fetch.fetch_if_missing(resolved[0][1])
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"auto-fetch failed: {e}")
    df = pd.read_parquet(resolved[0][1])
    df = df.loc[bt_module.VAL_START:bt_module.VAL_END]
    if len(df) < 100:
        raise HTTPException(400, "validation window has too few candles")

    is_crypto = "crypto" in str(resolved[0][1])
    BacktestClass = FractionalBacktest if is_crypto else Backtest
    bt = BacktestClass(
        df, UserStrategy,
        cash=bt_module.STARTING_CASH,
        commission=bt_module.COMMISSION,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run()
    eq = stats._equity_curve
    bh_series = df["Close"] / df["Close"].iloc[0] * float(eq["Equity"].iloc[0])
    max_dd = _safe_float(stats.get("Max. Drawdown [%]"))
    win_rate = _safe_float(stats.get("Win Rate [%]"))
    return {
        "timestamps": [t.isoformat() for t in eq.index],
        "equity": _safe_floats(eq["Equity"].astype(float).tolist()),
        "buy_and_hold": _safe_floats(bh_series.astype(float).tolist()),
        "drawdown": _safe_floats((-eq["DrawdownPct"].astype(float) * 100).tolist()),
        "metrics": {
            "sharpe": _safe_float(stats.get("Sharpe Ratio")),
            "sortino": _safe_float(stats.get("Sortino Ratio")),
            "max_drawdown": abs(max_dd) if max_dd is not None else None,
            "win_rate": (win_rate / 100.0) if win_rate is not None else None,
            "total_trades": int(stats.get("# Trades", 0) or 0),
            "total_return": _safe_float(stats.get("Return [%]")),
            "buy_and_hold_return": _safe_float(stats.get("Buy & Hold Return [%]")),
        },
    }


def _load_campaigns() -> dict:
    import tomllib
    with open(ROOT / "configs.toml", "rb") as f:
        return tomllib.load(f)


@app.get("/api/campaigns")
def get_campaigns() -> dict:
    """Return available campaigns and which is currently active."""
    try:
        available = list(_load_campaigns().keys())
    except Exception:
        available = ["stocks", "crypto"]
    return {"active": _active_campaign, "available": available}


@app.post("/api/campaign/{name}")
def set_campaign(name: str) -> dict:
    """Switch the active campaign and reload the backtest module."""
    global _active_campaign
    try:
        all_campaigns = _load_campaigns()
    except Exception:
        raise HTTPException(500, "configs.toml not found")
    if name not in all_campaigns:
        raise HTTPException(400, f"Unknown campaign: {name!r}. Available: {list(all_campaigns)}")
    if loop_proc.is_running():
        raise HTTPException(409, "Stop the running loop before switching campaign.")
    _active_campaign = name
    os.environ["CAMPAIGN"] = name  # backtest.py reads this on reload
    import importlib
    import backtest as _bt
    importlib.reload(_bt)
    wt = _worktree_for(name)
    return {
        "ok": True,
        "campaign": name,
        "worktree": str(wt),
        "worktree_ready": wt.exists(),
    }


@app.get("/api/setup")
def setup_status() -> dict:
    env_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    if not env_key and (ROOT / ".env").exists():
        env_key = "OPENROUTER_API_KEY" in (ROOT / ".env").read_text()

    files: list[dict] = []
    if DATA_DIR.exists():
        for f in sorted(DATA_DIR.rglob("*.parquet")):
            try:
                rows = len(pd.read_parquet(f))
            except Exception:
                rows = 0
            files.append({
                "path": str(f.relative_to(ROOT)).replace("\\", "/"),
                "rows": rows,
                "size_kb": f.stat().st_size // 1024,
            })

    # Check dirty state in the active campaign's worktree — that's the one
    # the loop will operate on. Falls back to ROOT if no campaign is selected.
    git_dirty = False
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_campaign_root(), capture_output=True, text=True, check=True,
        ).stdout
        git_dirty = bool(out.strip())
    except Exception:
        pass

    return {
        "checks": {
            "openrouter_api_key": env_key,
            "btc_data": (DATA_DIR / "crypto" / "BTC_USDT_4h.parquet").exists(),
            "venv_python": PYTHON.exists(),
            "strategy_file": _strategy_path().exists(),
            "program_md": PROGRAM.exists(),
            "git_clean": not git_dirty,
        },
        "data_files": files,
    }


# ────────────────────── action endpoints ─────────────────────────────

class BacktestRequest(BaseModel):
    window: str = "val"


@app.post("/api/backtest")
def run_backtest(req: BacktestRequest) -> dict:
    if req.window not in ("train", "val"):
        raise HTTPException(400, "window must be 'train' or 'val'")
    proc = subprocess.run(
        [str(PYTHON), "backtest.py", "--window", req.window],
        cwd=_campaign_root(), capture_output=True, text=True, timeout=300,
    )
    metrics: dict[str, float] = {}
    for m in re.finditer(r"^(\w+):\s+([-\d.]+)\s*$", proc.stdout, re.MULTILINE):
        try:
            metrics[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return {"stdout": proc.stdout, "stderr": proc.stderr, "exit_code": proc.returncode, "metrics": metrics}


class LoopStartRequest(BaseModel):
    iters: int = 5


@app.post("/api/loop/start")
def loop_start(req: LoopStartRequest) -> dict:
    if loop_proc.is_running():
        raise HTTPException(409, "loop already running")
    if not (1 <= req.iters <= 500):
        raise HTTPException(400, "iters must be 1..500")
    campaign = _campaign_or_400()
    try:
        loop_proc.start(req.iters, campaign)
    except RuntimeError as e:
        # Surface worktree-missing as 409 so the UI can offer guidance.
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"started": True, "iters": req.iters, "campaign": campaign}


@app.post("/api/loop/stop")
def loop_stop() -> dict:
    return {"stopped": loop_proc.stop()}


@app.get("/api/loop/status")
def loop_status() -> dict:
    return {"running": loop_proc.is_running(), "buffered_lines": len(loop_proc.buffer)}


@app.get("/api/loop/stream")
async def loop_stream():
    async def event_gen():
        # Track the highest line-id we've already sent. Monotonic IDs survive
        # buffer.clear() between runs, so an SSE reconnect can never re-emit
        # lines that were already pushed to the client (no duplicate console).
        last_id = loop_proc.next_id - 1
        while True:
            new_lines = [(lid, txt) for lid, txt in list(loop_proc.buffer) if lid > last_id]
            if new_lines:
                for lid, line in new_lines:
                    yield {"event": "line", "data": json.dumps({"line": line})}
                last_id = new_lines[-1][0]
            yield {"event": "status", "data": json.dumps({
                "running": loop_proc.is_running(),
                "lines": loop_proc.next_id,
                "current_iter": loop_proc.current_iter,
                "total_iters": loop_proc.total_iters,
            })}
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_gen())


# ───────────────────── paper-log / alpaca routes ─────────────────────

PAPER_LOG = ROOT / "results" / "paper.log"


# ── strategy version tracking (Trading tab — Phase 1) ──
# A "version" is a commit on `main` that touched strategies/<campaign>.py.
# These are the strategies that actually went live (sync_branches.py is the only
# writer on main). Newest-first; the head is the currently deployed version.

def _git_log_versions(campaign: str) -> list[dict]:
    rel = f"strategies/{campaign}.py"
    cmd = [
        "git", "-C", str(ROOT), "log", "main", "--follow",
        "--format=%H%x09%cI%x09%s", "--", rel,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return []

    rows: list[dict] = []
    for line in out.strip().splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        commit, promoted_at, subject = parts
        rows.append({
            "commit": commit,
            "short": commit[:7],
            "promoted_at": promoted_at,
            "subject": subject,
        })

    # rows are newest-first. retired_at[i] = promoted_at[i-1]; head has retired_at=None.
    for i, r in enumerate(rows):
        r["retired_at"] = rows[i - 1]["promoted_at"] if i > 0 else None
    return rows


@app.get("/api/strategy/versions")
def strategy_versions(campaign: str) -> dict:
    """Promotion history for a campaign — every commit on main that changed
    strategies/<campaign>.py. The first entry is the currently deployed version."""
    if campaign not in ("stocks", "crypto"):
        raise HTTPException(400, "campaign must be 'stocks' or 'crypto'")
    versions = _git_log_versions(campaign)
    return {"campaign": campaign, "versions": versions}


# ── paper trading equity curve (Phase 1) ──
# Walks results/paper.log, pairs BUY→SELL per symbol, accumulates realized
# P&L into a time-series + per-trade ledger. Also returns one point per
# closed trade so the frontend can plot a step curve.

def _paper_equity(campaign: str) -> dict:
    asset_filter = "crypto" if campaign == "crypto" else "stock"
    if not PAPER_LOG.exists():
        return {"trades": [], "points": [], "summary": {"total_trades": 0, "total_pnl": 0.0, "wins": 0, "losses": 0}}

    records: list[dict] = []
    with PAPER_LOG.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("asset") != asset_filter:
                continue
            records.append(rec)

    # Chronological order — paper.log is append-only, but be defensive.
    records.sort(key=lambda r: r.get("ts", ""))

    open_legs: dict[str, dict] = {}  # symbol → {entry_price, qty, ts}
    trades: list[dict] = []
    cumulative = 0.0

    def _is_buy(action: str) -> bool:
        return "buy" in (action or "")  # captures buy + dry-buy

    def _is_sell(action: str) -> bool:
        return "sell" in (action or "")

    for rec in records:
        sym = rec.get("symbol")
        ts = rec.get("ts")
        scan = rec.get("scan") or {}
        ex = rec.get("execution") or {}
        action = ex.get("action") or ""

        if _is_buy(action) and sym not in open_legs:
            entry = scan.get("entry_price") or scan.get("last_close")
            if entry is None:
                continue
            qty_raw = ex.get("qty") or ex.get("would_qty")
            try:
                qty = float(qty_raw) if isinstance(qty_raw, (int, float, str)) and str(qty_raw).replace('.', '', 1).isdigit() else 1.0
            except (ValueError, TypeError):
                qty = 1.0
            open_legs[sym] = {"entry_price": float(entry), "qty": qty, "entry_ts": ts}

        elif _is_sell(action) and sym in open_legs:
            leg = open_legs.pop(sym)
            exit_price = scan.get("exit_price") or scan.get("last_close")
            if exit_price is None:
                continue
            pnl = (float(exit_price) - leg["entry_price"]) * leg["qty"]
            cumulative += pnl
            trades.append({
                "ts": ts,
                "symbol": sym,
                "entry_ts": leg["entry_ts"],
                "entry_price": leg["entry_price"],
                "exit_price": float(exit_price),
                "qty": leg["qty"],
                "realized_pnl": pnl,
                "cumulative_pnl": cumulative,
            })

    points = [{"ts": t["ts"], "cumulative_pnl": t["cumulative_pnl"]} for t in trades]
    wins = sum(1 for t in trades if t["realized_pnl"] > 0)
    losses = sum(1 for t in trades if t["realized_pnl"] < 0)
    return {
        "trades": trades,
        "points": points,
        "open_positions": [
            {"symbol": s, **leg} for s, leg in open_legs.items()
        ],
        "summary": {
            "total_trades": len(trades),
            "total_pnl": cumulative,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / len(trades)) if trades else 0.0,
        },
    }


@app.get("/api/paper/equity")
def paper_equity(campaign: str) -> dict:
    """Realized paper-trading P&L curve + ledger for the campaign.
    Reconstructed from results/paper.log by pairing BUY→SELL per symbol."""
    if campaign not in ("stocks", "crypto"):
        raise HTTPException(400, "campaign must be 'stocks' or 'crypto'")
    return {"campaign": campaign, **_paper_equity(campaign)}


# ── live chart data (Phase 2) ──
# Fetches recent bars, runs the current strategy on them, extracts
# indicator series via introspection, and returns price + indicators +
# entry/exit markers. The "auto-detect indicators" trick: backtesting.py
# stores every self.I() registration on `strategy._indicators`. We pull
# those back out and filter to ones that overlay sensibly on the price
# axis (median-value heuristic). When the loop swaps to a different
# strategy with different indicators tomorrow, the chart adapts with
# zero code changes.

import time as _time
_chart_cache: dict[tuple[str, str], tuple[float, dict]] = {}
_CHART_TTL = 90.0  # seconds — bars only roll over every 4h/1d


def _default_symbol(campaign: str) -> str:
    if campaign == "crypto":
        return (os.environ.get("PAPER_CRYPTO_WATCHLIST") or "BTC/USD").split(",")[0].strip()
    return (os.environ.get("PAPER_WATCHLIST") or "SPY,QQQ,TSLA,NVDA").split(",")[0].strip()


@app.get("/api/paper/chart-data")
def paper_chart_data(campaign: str, symbol: str | None = None, bars: int = 80) -> dict:
    """Recent bars + strategy indicators (auto-detected) + paper-trade markers.

    Cached for 90s so the dashboard's 30s refresh doesn't hammer Alpaca/yfinance.
    """
    if campaign not in ("stocks", "crypto"):
        raise HTTPException(400, "campaign must be 'stocks' or 'crypto'")

    sym = (symbol or _default_symbol(campaign)).strip()
    asset = "crypto" if campaign == "crypto" else "stock"

    cache_key = (asset, sym)
    cached = _chart_cache.get(cache_key)
    if cached and (_time.time() - cached[0] < _CHART_TTL):
        return cached[1]

    import numpy as np
    try:
        from live_trade import _fetch_bars  # reuses crypto/stock fetchers
        from backtesting import Backtest
        from backtesting.lib import FractionalBacktest
        from backtest import load_strategy_class
    except Exception as e:
        raise HTTPException(500, f"import failure: {e}")

    timeframe = os.environ.get("PAPER_CRYPTO_TIMEFRAME", "4h") if asset == "crypto" else "1d"
    # Calendar days to request — generous enough that we get >= `bars` rows after warmup.
    days = max(60, bars * 5 if asset != "crypto" else bars)

    try:
        df_full = _fetch_bars(sym, asset, days=days, timeframe=timeframe)
    except Exception as e:
        raise HTTPException(502, f"data fetch failed: {e}")

    if len(df_full) < 30:
        raise HTTPException(502, f"only {len(df_full)} bars returned for {sym}")

    strategy_file = f"strategies/{campaign}.py"
    try:
        UserStrategy = load_strategy_class(strategy_file)
    except Exception as e:
        raise HTTPException(500, f"load strategy {strategy_file} failed: {e}")

    # Match the real paper trader's per-symbol cash budget so simulated dollar
    # P&L is directly comparable to actual paper trades. FractionalBacktest
    # for crypto (lets us "buy" fractions of a BTC at $10K cash); regular
    # Backtest for stocks (whole-share, plenty at this notional).
    cash = float(os.environ.get("PAPER_PER_SYMBOL_CASH", 10000))
    commission = float(os.environ.get("COMMISSION", 0.001))
    BacktestClass = FractionalBacktest if asset == "crypto" else Backtest
    try:
        bt = BacktestClass(df_full, UserStrategy, cash=cash, commission=commission,
                           exclusive_orders=True, finalize_trades=True)
        stats = bt.run()
    except Exception as e:
        raise HTTPException(500, f"backtest failed: {e}")

    strat = stats._strategy

    # Slice to the last `bars` rows for display.
    n = len(df_full)
    start = max(0, n - bars)
    df = df_full.iloc[start:]

    bars_out = [{
        "ts": ts.isoformat(),
        "o": float(row.Open), "h": float(row.High),
        "l": float(row.Low), "c": float(row.Close),
        "v": float(row.Volume),
    } for ts, row in df.iterrows()]

    # Build simulated trades + cumulative P&L within the visible window.
    # This answers the question that actually matters: "if the deployed
    # strategy had been live over these recent bars, would it have made money?"
    # Until paper.log has real trades, this is the closest signal of strategy
    # health we have on real recent market data (not stale 2020-2024 backtest).
    sim_markers: list[dict] = []
    sim_trades: list[dict] = []
    sim_pnl_curve: list[dict] = []  # one point per visible bar; cumulative realized
    cumulative_pnl = 0.0
    closed_trades_in_window = 0
    wins = 0

    trades_df = stats.get("_trades")
    window_start = df.index[0]
    realized_at: dict[str, float] = {}  # exit_ts_iso → cumulative_pnl_after
    if trades_df is not None and len(trades_df):
        for _, t in trades_df.iterrows():
            et = pd.Timestamp(t["EntryTime"])
            xt = pd.Timestamp(t["ExitTime"]) if pd.notna(t.get("ExitTime")) else None

            entry_in_window = et >= window_start
            exit_in_window = xt is not None and xt >= window_start

            if entry_in_window:
                sim_markers.append({"ts": et.isoformat(), "type": "BUY", "price": float(t["EntryPrice"])})
            if exit_in_window:
                sim_markers.append({"ts": xt.isoformat(), "type": "SELL", "price": float(t["ExitPrice"])})

            # Count realized P&L only for trades that closed within the visible
            # window; in-progress trades don't contribute until they exit.
            if exit_in_window:
                pnl = float(t["PnL"])
                cumulative_pnl += pnl
                closed_trades_in_window += 1
                if pnl > 0:
                    wins += 1
                sim_trades.append({
                    "entry_ts": et.isoformat(),
                    "exit_ts": xt.isoformat(),
                    "entry_price": float(t["EntryPrice"]),
                    "exit_price": float(t["ExitPrice"]),
                    "pnl": pnl,
                    "return_pct": float(t["ReturnPct"]) * 100.0,
                })
                realized_at[xt.isoformat()] = cumulative_pnl

    # Step curve: P&L only changes at exit timestamps; in between it's flat.
    running = 0.0
    for ts in df.index:
        running = realized_at.get(ts.isoformat(), running)
        sim_pnl_curve.append({"ts": ts.isoformat(), "cumulative_pnl": running})

    # Real paper.log markers for THIS symbol — these are the only "real" signals
    # (ones the executor actually saw on a live bar close).
    real_markers: list[dict] = []
    if PAPER_LOG.exists():
        with PAPER_LOG.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("symbol") != sym:
                    continue
                scan = rec.get("scan") or {}
                if scan.get("signal") == "BUY" and scan.get("entry_price") and scan.get("entry_time"):
                    real_markers.append({"ts": scan["entry_time"], "type": "BUY", "price": float(scan["entry_price"])})
                elif scan.get("signal") == "SELL" and scan.get("exit_price") and scan.get("exit_time"):
                    real_markers.append({"ts": scan["exit_time"], "type": "SELL", "price": float(scan["exit_price"])})

    out = {
        "campaign": campaign,
        "symbol": sym,
        "asset": asset,
        "timeframe": timeframe,
        "bars": bars_out,
        "sim_markers": sim_markers,
        "real_markers": real_markers,
        "sim_pnl_curve": sim_pnl_curve,
        "sim_pnl_total": cumulative_pnl,
        "sim_trades": sim_trades,
        "sim_trades_count": closed_trades_in_window,
        "sim_win_rate": (wins / closed_trades_in_window) if closed_trades_in_window else None,
        "last_bar": {
            "ts": df.index[-1].isoformat(),
            "close": float(df["Close"].iloc[-1]),
            "in_position": bool(strat.position),
        },
        "watchlist": [s.strip() for s in (
            os.environ.get("PAPER_CRYPTO_WATCHLIST" if asset == "crypto" else "PAPER_WATCHLIST", "")
        ).split(",") if s.strip()] or [sym],
    }
    _chart_cache[cache_key] = (_time.time(), out)
    return out


# ── per-version performance + anomaly stats (Phases 4 + 5) ──

@app.get("/api/paper/version-stats")
def paper_version_stats(campaign: str) -> dict:
    """Slice paper trades by which strategy version was deployed when each
    trade fired. Returns one row per version, newest-first, with per-version
    realized stats. Versions with zero trades still appear (so the user can
    see "this version was deployed but hasn't traded yet")."""
    if campaign not in ("stocks", "crypto"):
        raise HTTPException(400, "campaign must be 'stocks' or 'crypto'")

    versions = _git_log_versions(campaign)
    eq = _paper_equity(campaign)
    trades = eq["trades"]

    def _ver_for(ts: str) -> str | None:
        # The version active at time ts is the newest version with promoted_at <= ts.
        for v in versions:  # newest first
            if v["promoted_at"] <= ts:
                return v["commit"]
        return None

    by_version: dict[str, list[dict]] = {}
    for t in trades:
        cv = _ver_for(t["ts"])
        if cv is None:
            continue
        by_version.setdefault(cv, []).append(t)

    rows: list[dict] = []
    for v in versions:
        v_trades = by_version.get(v["commit"], [])
        pnl = sum(t["realized_pnl"] for t in v_trades)
        wins = sum(1 for t in v_trades if t["realized_pnl"] > 0)
        rows.append({
            "commit": v["commit"],
            "short": v["short"],
            "promoted_at": v["promoted_at"],
            "retired_at": v["retired_at"],
            "subject": v["subject"],
            "trades": len(v_trades),
            "pnl": pnl,
            "wins": wins,
            "losses": sum(1 for t in v_trades if t["realized_pnl"] < 0),
            "win_rate": (wins / len(v_trades)) if v_trades else None,
        })
    return {"campaign": campaign, "versions": rows}


@app.get("/api/paper/anomalies")
def paper_anomalies(campaign: str) -> dict:
    """Lightweight diagnostic stats — currently just 'days since last signal'
    for the current version, which catches a stuck/silent strategy fast."""
    if campaign not in ("stocks", "crypto"):
        raise HTTPException(400, "campaign must be 'stocks' or 'crypto'")

    if not PAPER_LOG.exists():
        return {"days_since_last_signal": None, "last_signal_ts": None, "last_signal_symbol": None}

    asset_filter = "crypto" if campaign == "crypto" else "stock"
    last_ts: str | None = None
    last_sym: str | None = None
    last_sig: str | None = None
    with PAPER_LOG.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("asset") != asset_filter:
                continue
            scan = rec.get("scan") or {}
            if scan.get("signal") in ("BUY", "SELL"):
                ts = rec.get("ts")
                if ts and (last_ts is None or ts > last_ts):
                    last_ts, last_sym, last_sig = ts, rec.get("symbol"), scan["signal"]

    days = None
    if last_ts:
        from datetime import datetime as _dt, timezone as _tz
        try:
            dt = _dt.fromisoformat(last_ts.replace("Z", "+00:00"))
            days = (_dt.now(_tz.utc) - dt).total_seconds() / 86400.0
        except ValueError:
            pass

    return {
        "days_since_last_signal": days,
        "last_signal_ts": last_ts,
        "last_signal_symbol": last_sym,
        "last_signal_kind": last_sig,
    }


@app.get("/api/paper-log")
def paper_log_route(limit: int = 200) -> list[dict]:
    """Return the last `limit` records from results/paper.log, newest first."""
    if not PAPER_LOG.exists():
        return []
    records: list[dict] = []
    try:
        with PAPER_LOG.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception:
        return []
    return list(reversed(records[-limit:]))


@app.get("/api/alpaca/positions")
def alpaca_positions() -> dict:
    """Fetch open positions + account equity from Alpaca.
    Returns {"error": "..."} if credentials are missing or alpaca-py not installed."""
    api_key = os.environ.get("ALPACA_API_KEY", "").strip()
    api_secret = os.environ.get("ALPACA_API_SECRET", "").strip()
    if not api_key or not api_secret:
        return {"error": "ALPACA_API_KEY / ALPACA_API_SECRET not set in .env"}
    paper = os.environ.get("ALPACA_PAPER", "True").lower() != "false"
    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        return {"error": "alpaca-py not installed (pip install alpaca-py)"}
    try:
        client = TradingClient(api_key, api_secret, paper=paper)
        account = client.get_account()
        raw_positions = client.get_all_positions()
        positions = []
        for p in raw_positions:
            positions.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price) if p.current_price else None,
                "market_value": float(p.market_value) if p.market_value else None,
                "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl else None,
                "unrealized_plpc": float(p.unrealized_plpc) if p.unrealized_plpc else None,
                "side": str(p.side),
            })
        return {
            "account": {
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "paper": paper,
                "status": str(account.status),
            },
            "positions": positions,
        }
    except Exception as e:
        return {"error": str(e)}


class PaperTradeRequest(BaseModel):
    asset: str = "stock"
    symbols: str = ""
    dry: bool = True
    live: bool = False


@app.post("/api/paper-trade/start")
def paper_trade_start(req: PaperTradeRequest) -> dict:
    if req.asset not in ("stock", "crypto"):
        raise HTTPException(400, "asset must be 'stock' or 'crypto'")
    try:
        paper_proc.start(
            asset=req.asset,
            symbols=req.symbols.strip() or None,
            dry=req.dry,
            live=req.live,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"started": True}


@app.post("/api/paper-trade/stop")
def paper_trade_stop() -> dict:
    return {"stopped": paper_proc.stop()}


@app.get("/api/paper-trade/status")
def paper_trade_status() -> dict:
    return {"running": paper_proc.is_running(), "buffered_lines": len(paper_proc.buffer)}


@app.get("/api/paper-trade/stream")
async def paper_trade_stream():
    async def event_gen():
        last_id = paper_proc.next_id - 1
        while True:
            new_lines = [(lid, txt) for lid, txt in list(paper_proc.buffer) if lid > last_id]
            if new_lines:
                for lid, line in new_lines:
                    yield {"event": "line", "data": json.dumps({"line": line})}
                last_id = new_lines[-1][0]
            yield {"event": "status", "data": json.dumps({
                "running": paper_proc.is_running(),
                "lines": paper_proc.next_id,
            })}
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_gen())


# ─────────────────────────── entry point ─────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("\n  TradingBot dashboard -> http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

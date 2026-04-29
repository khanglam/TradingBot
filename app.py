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

ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "web" / "index.html"
RESULTS = ROOT / "results" / "results.tsv"
STRATEGY = ROOT / "strategy.py"
PROGRAM = ROOT / "program.md"
DATA_DIR = ROOT / "data"

PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

app = FastAPI(title="TradingBot Autoresearch UI")


# ───────────────────────────── helpers ───────────────────────────────

_NEW_SCHEMA_COLS = [
    "commit", "val_sharpe", "sortino", "sharpe_ann_4h", "calmar", "psr", "dsr",
    "skew", "kurtosis", "max_drawdown", "win_rate", "total_trades",
    "status", "description",
]


def _load_results() -> pd.DataFrame:
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
    """Read CLAUDE_MODEL env override; else regex DEFAULT_MODEL out of loop.py.
    Avoids importing loop (which has init-time side effects)."""
    if env := os.environ.get("CLAUDE_MODEL"):
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

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _append(self, line: str) -> None:
        self.buffer.append((self.next_id, line))
        self.next_id += 1

    def start(self, iters: int) -> None:
        with self.lock:
            if self.is_running():
                raise RuntimeError("Loop already running.")
            self.buffer.clear()
            self._append(f"[ui] starting loop.py --iters {iters}")
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            self.proc = subprocess.Popen(
                [str(PYTHON), "loop.py", "--iters", str(int(iters))],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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
    return {
        "best_sharpe": best,
        "total": int(len(df)),
        "keeps": int(len(keeps)),
        "discards": int(len(discards)),
        "crashes": int(len(crashes)),
        "keep_rate": (len(keeps) / len(df)) if len(df) else 0.0,
        "latest": latest,
        "model": _active_model(),
    }


@app.get("/api/results")
def results() -> list[dict[str, Any]]:
    return _df_to_records(_load_results())


@app.get("/api/strategy")
def strategy_source() -> dict:
    if not STRATEGY.exists():
        raise HTTPException(404, "strategy.py not found")
    return {"source": STRATEGY.read_text(encoding="utf-8")}


@app.get("/api/program")
def program_source() -> dict:
    if not PROGRAM.exists():
        raise HTTPException(404, "program.md not found")
    return {"source": PROGRAM.read_text(encoding="utf-8")}


@app.get("/api/git-log")
def git_log(n: int = 20) -> dict:
    try:
        out = subprocess.run(
            ["git", "log", "--pretty=format:%h|%ar|%s", f"-{int(n)}", "--", "strategy.py"],
            cwd=ROOT, capture_output=True, text=True, check=True,
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
def equity_curve():
    """Run strategy.py in-process to produce equity + drawdown for charting."""
    try:
        import importlib

        sys.path.insert(0, str(ROOT))
        import backtest as bt_module
        importlib.reload(bt_module)
        if "strategy" in sys.modules:
            importlib.reload(sys.modules["strategy"])
        from backtesting import Backtest
        from strategy import Strategy as UserStrategy

        # Equity curve is single-asset by definition: pick the first symbol
        # the harness would resolve from $SYMBOLS (or DEFAULT_SYMBOLS).
        spec = os.environ.get("SYMBOLS") or bt_module.DEFAULT_SYMBOLS
        resolved = bt_module._resolve_symbols(spec)
        if not resolved:
            raise HTTPException(400, "no symbols resolved from $SYMBOLS")
        df = pd.read_parquet(resolved[0][1])
        df = df.loc[bt_module.VAL_START:bt_module.VAL_END]
        if len(df) < 100:
            raise HTTPException(400, "validation window has too few candles")

        bt = Backtest(
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
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/setup")
def setup_status() -> dict:
    env_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not env_key and (ROOT / ".env").exists():
        env_key = "ANTHROPIC_API_KEY" in (ROOT / ".env").read_text()

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

    git_dirty = False
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
        git_dirty = bool(out.strip())
    except Exception:
        pass

    return {
        "checks": {
            "anthropic_api_key": env_key,
            "btc_data": (DATA_DIR / "crypto" / "BTC_USDT_4h.parquet").exists(),
            "venv_python": PYTHON.exists(),
            "strategy_py": STRATEGY.exists(),
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
        cwd=ROOT, capture_output=True, text=True, timeout=300,
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
    try:
        loop_proc.start(req.iters)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"started": True, "iters": req.iters}


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
            })}
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_gen())


# ─────────────────────────── entry point ─────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("\n  TradingBot dashboard -> http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

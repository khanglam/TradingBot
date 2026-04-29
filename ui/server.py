"""FastAPI backend for the TradingBot dashboard.

Run:
    uvicorn ui.server:app --reload
or double-click run_ui.bat / run_ui.sh.

Routes:
    GET  /                      → static/index.html
    GET  /api/health            → ping
    GET  /api/summary           → KPI snapshot (best sharpe, counts, latest)
    GET  /api/results           → results.tsv as JSON array
    GET  /api/strategy          → strategy.py source
    GET  /api/program           → program.md source
    GET  /api/equity            → equity curve + drawdown for current strategy
    GET  /api/git-log?n=20      → git history of strategy.py
    GET  /api/setup             → environment / data file check
    POST /api/backtest          → run backtest.py once, return parsed metrics
    POST /api/loop/start        → spawn loop.py in background
    POST /api/loop/stop         → kill the running loop
    GET  /api/loop/status       → is the loop running?
    GET  /api/loop/stream       → SSE stream of live stdout
"""
from __future__ import annotations

import asyncio
import json
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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
RESULTS = ROOT / "results.tsv"
STRATEGY = ROOT / "strategy.py"
PROGRAM = ROOT / "program.md"
DATA_DIR = ROOT / "data"

PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

app = FastAPI(title="TradingBot Autoresearch UI")


# ─────────────────────────── helpers ─────────────────────────────────

def _load_results() -> pd.DataFrame:
    if not RESULTS.exists():
        return pd.DataFrame(columns=[
            "commit", "val_sharpe", "max_drawdown",
            "win_rate", "total_trades", "status", "description",
        ])
    df = pd.read_csv(RESULTS, sep="\t")
    for col in ("val_sharpe", "max_drawdown", "win_rate"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["total_trades"] = pd.to_numeric(df["total_trades"], errors="coerce").fillna(0).astype(int)
    return df


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")


# ───────────────────────── loop process state ────────────────────────

class LoopProc:
    """Singleton wrapper around the running loop.py subprocess."""
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.buffer: deque[str] = deque(maxlen=2000)
        self.lock = threading.Lock()
        self.reader_thread: threading.Thread | None = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, iters: int) -> None:
        with self.lock:
            if self.is_running():
                raise RuntimeError("Loop already running.")
            self.buffer.clear()
            self.buffer.append(f"[ui] starting loop.py --iters {iters}")
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            self.proc = subprocess.Popen(
                [str(PYTHON), "loop.py", "--iters", str(int(iters))],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=creationflags,
            )
            self.reader_thread = threading.Thread(target=self._drain, daemon=True)
            self.reader_thread.start()

    def _drain(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in iter(self.proc.stdout.readline, ""):
            self.buffer.append(line.rstrip("\n"))
        rc = self.proc.wait()
        self.buffer.append(f"[ui] loop exited with code {rc}")

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
            self.buffer.append("[ui] loop stopped by user")
            return True


loop_proc = LoopProc()


# ─────────────────────────── routes ──────────────────────────────────

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
    }


@app.get("/api/results")
def results() -> list[dict[str, Any]]:
    df = _load_results()
    return _df_to_records(df)


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
def equity_curve() -> dict:
    """Run strategy.py in-process to produce equity + drawdown for charting.

    Same window/cash/commission as backtest.py — the metrics will agree.
    This is a viz endpoint, not the official oracle.
    """
    try:
        import importlib

        sys.path.insert(0, str(ROOT))
        import backtest as bt_module
        importlib.reload(bt_module)
        if "strategy" in sys.modules:
            importlib.reload(sys.modules["strategy"])
        from backtesting import Backtest
        from strategy import Strategy as UserStrategy

        df = pd.read_parquet(ROOT / bt_module.DEFAULT_DATA)
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
        equity = eq["Equity"].astype(float).tolist()
        drawdown = (-eq["DrawdownPct"].astype(float) * 100).tolist()
        timestamps = [t.isoformat() for t in eq.index]
        close = df["Close"].astype(float).tolist()
        bh_scaled = (df["Close"] / df["Close"].iloc[0] * equity[0]).astype(float).tolist()

        return {
            "timestamps": timestamps,
            "equity": equity,
            "buy_and_hold": bh_scaled,
            "drawdown": drawdown,
            "close": close,
            "metrics": {
                "sharpe": float(stats.get("Sharpe Ratio", 0.0) or 0.0),
                "sortino": float(stats.get("Sortino Ratio", 0.0) or 0.0),
                "max_drawdown": abs(float(stats.get("Max. Drawdown [%]", 0.0) or 0.0)),
                "win_rate": float(stats.get("Win Rate [%]", 0.0) or 0.0) / 100.0,
                "total_trades": int(stats.get("# Trades", 0) or 0),
                "total_return": float(stats.get("Return [%]", 0.0) or 0.0),
                "buy_and_hold_return": float(stats.get("Buy & Hold Return [%]", 0.0) or 0.0),
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


# ──────────────────── action endpoints ───────────────────────────────

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
    out = proc.stdout
    metrics = {}
    pat = re.compile(r"^(\w+):\s+([-\d.]+)\s*$", re.MULTILINE)
    for m in pat.finditer(out):
        try:
            metrics[m.group(1)] = float(m.group(2))
        except ValueError:
            continue
    return {
        "stdout": out,
        "stderr": proc.stderr,
        "exit_code": proc.returncode,
        "metrics": metrics,
    }


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
    stopped = loop_proc.stop()
    return {"stopped": stopped}


@app.get("/api/loop/status")
def loop_status() -> dict:
    return {
        "running": loop_proc.is_running(),
        "buffered_lines": len(loop_proc.buffer),
    }


@app.get("/api/loop/stream")
async def loop_stream():
    async def event_gen():
        last_len = 0
        while True:
            buf = list(loop_proc.buffer)
            if len(buf) > last_len:
                for line in buf[last_len:]:
                    yield {"event": "line", "data": json.dumps({"line": line})}
                last_len = len(buf)
            yield {
                "event": "status",
                "data": json.dumps({
                    "running": loop_proc.is_running(),
                    "lines": len(buf),
                }),
            }
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_gen())


# ──────────────────────── static / index ─────────────────────────────

app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


def main() -> None:
    import uvicorn
    uvicorn.run("ui.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()

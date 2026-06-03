#!/usr/bin/env python3
"""Stop local TradingBot loop/backtest processes. Used by /run-loop skill."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # repo root: .../TradingBot
PATTERNS = ("loop.py", "backtest.py")


def _repo_root() -> Path:
    p = ROOT
    for _ in range(6):
        if (p / "loop.py").exists():
            return p
        p = p.parent
    return ROOT


def main() -> int:
    root = _repo_root()
    root_s = str(root).lower().replace("\\", "/")
    killed: list[int] = []

    if sys.platform == "win32":
        ps = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
        )
        import json
        try:
            rows = json.loads(ps.stdout) if ps.stdout.strip() else []
            if isinstance(rows, dict):
                rows = [rows]
        except json.JSONDecodeError:
            rows = []
        for row in rows:
            cmd = (row.get("CommandLine") or "").replace("\\", "/").lower()
            pid = row.get("ProcessId")
            if pid and root_s in cmd and any(x in cmd for x in PATTERNS):
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False)
                killed.append(int(pid))
    else:
        out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "python" not in line.lower():
                continue
            if root_s not in line.replace("\\", "/").lower():
                continue
            if not any(x in line for x in PATTERNS):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            pid = int(parts[1])
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)

    if killed:
        print(f"[stop_loops] stopped PIDs: {', '.join(map(str, killed))}")
    else:
        print("[stop_loops] no TradingBot loop/backtest processes found")
    return 0


if __name__ == "__main__":
    sys.exit(main())

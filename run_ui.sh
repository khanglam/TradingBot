#!/usr/bin/env bash
# One-click launcher for the TradingBot dashboard UI.
# Open http://localhost:8000 after this starts.
cd "$(dirname "$0")"
./.venv/Scripts/python.exe -m uvicorn ui.server:app --host 127.0.0.1 --port 8000

@echo off
REM One-click launcher for the TradingBot dashboard UI.
REM Open http://localhost:8000 after this starts.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m uvicorn ui.server:app --host 127.0.0.1 --port 8000

@echo off
REM One-click launcher for the TradingBot dashboard.
REM Opens http://localhost:8501 in your default browser.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run dashboard.py

#!/usr/bin/env bash
# One-click launcher for the TradingBot dashboard.
cd "$(dirname "$0")"
./.venv/Scripts/python.exe -m streamlit run dashboard.py

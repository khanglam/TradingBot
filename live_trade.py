"""Paper / live trader using LumiBot, mirrors the logic in strategy.py.

This is the deployment surface. The autoresearch loop NEVER touches this file —
it only optimizes strategy.py. When a candidate strategy is ready for paper
trading, you flip PAPER_TRADING via env var or CLI flag and run this.

Setup once:
    1. Create an Alpaca account (free) → paper API keys
       https://app.alpaca.markets/paper/dashboard/overview
    2. .env file at project root:
           ALPACA_API_KEY=...
           ALPACA_API_SECRET=...
           ALPACA_PAPER=True              # flip to False to go live
           ANTHROPIC_API_KEY=...           # only used by loop.py

Usage:
    python live_trade.py                  # paper-trade SPY using strategy.py
    python live_trade.py --symbol BTC/USD --asset crypto
    python live_trade.py --backtest       # quick LumiBot backtest sanity check

Note: We re-implement the EMA crossover here because LumiBot's bar API differs
from backtesting.py's. If you change strategy.py to use a different signal,
mirror it here. The autoresearch loop's evaluation harness remains backtest.py;
this file is just the paper/live execution layer.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _load_strategy_params() -> dict:
    """Read fast/slow defaults from the current strategy.py if present."""
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    try:
        mod = importlib.import_module("strategy")
        S = mod.Strategy
        return {"fast": int(getattr(S, "fast", 20)), "slow": int(getattr(S, "slow", 50))}
    except Exception:
        return {"fast": 20, "slow": 50}


def build_strategy_class():
    """Construct a LumiBot Strategy that mirrors the EMA crossover in strategy.py."""
    from lumibot.strategies.strategy import Strategy as LBStrategy

    params = _load_strategy_params()

    class EmaCrossover(LBStrategy):
        parameters = {
            "symbol": "SPY",
            "fast": params["fast"],
            "slow": params["slow"],
            "asset_type": "stock",  # "stock" or "crypto"
        }

        def initialize(self):
            self.sleeptime = "1D"
            self.set_market("NYSE")

        def on_trading_iteration(self):
            symbol = self.parameters["symbol"]
            fast = self.parameters["fast"]
            slow = self.parameters["slow"]

            bars = self.get_historical_prices(symbol, slow + 5, "day")
            if bars is None or len(bars.df) < slow + 2:
                return

            close = bars.df["close"]
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()

            crossed_up = ema_fast.iloc[-2] <= ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1]
            crossed_dn = ema_fast.iloc[-2] >= ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1]

            position = self.get_position(symbol)
            cash = self.cash

            if crossed_up and position is None:
                price = self.get_last_price(symbol)
                qty = int(cash // price)
                if qty > 0:
                    order = self.create_order(symbol, qty, "buy")
                    self.submit_order(order)
                    self.log_message(f"BUY {qty} {symbol} @ ~{price}")
            elif crossed_dn and position is not None and position.quantity > 0:
                order = self.create_order(symbol, position.quantity, "sell")
                self.submit_order(order)
                self.log_message(f"SELL {position.quantity} {symbol}")

    return EmaCrossover


def run_live(symbol: str, asset_type: str, paper: bool) -> None:
    from lumibot.brokers import Alpaca
    from lumibot.traders import Trader

    if not os.environ.get("ALPACA_API_KEY") or not os.environ.get("ALPACA_API_SECRET"):
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_API_SECRET (use the paper keys for paper trading).")

    config = {
        "API_KEY": os.environ["ALPACA_API_KEY"],
        "API_SECRET": os.environ["ALPACA_API_SECRET"],
        "PAPER": paper,
    }
    broker = Alpaca(config)
    StratCls = build_strategy_class()
    strategy = StratCls(broker=broker, parameters={"symbol": symbol, "asset_type": asset_type})

    trader = Trader()
    trader.add_strategy(strategy)
    print(f"[live_trade] starting {'PAPER' if paper else 'LIVE'} session: {symbol} ({asset_type})")
    trader.run_all()


def run_quick_backtest(symbol: str = "SPY") -> None:
    """Quick smoke test using LumiBot's yahoo backtest source."""
    from lumibot.backtesting import YahooDataBacktesting

    StratCls = build_strategy_class()
    end = datetime.now() - timedelta(days=1)
    start = end - timedelta(days=365 * 3)
    StratCls.run_backtest(
        YahooDataBacktesting,
        start,
        end,
        parameters={"symbol": symbol, "asset_type": "stock"},
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--asset", choices=["stock", "crypto"], default="stock")
    p.add_argument("--live", action="store_true", help="trade real money (default is paper)")
    p.add_argument("--backtest", action="store_true", help="quick LumiBot backtest sanity check")
    args = p.parse_args()

    if args.backtest:
        run_quick_backtest(args.symbol)
        return

    paper_env = os.environ.get("ALPACA_PAPER", "True").lower() != "false"
    paper = paper_env and not args.live
    run_live(args.symbol, args.asset, paper)


if __name__ == "__main__":
    main()

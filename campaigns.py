"""Campaign configuration profiles.

Each campaign maps to a dict of settings that override backtest.py defaults.
Switch locally with CAMPAIGN=crypto or CAMPAIGN=stocks in .env.
CI sets CAMPAIGN from the matrix name automatically.

Individual env vars (SYMBOLS, VAL_START, etc.) still override these values:
    CAMPAIGN=crypto VAL_START=2021-01-01 python loop.py --iters 5
"""

CAMPAIGNS: dict[str, dict] = {
    "crypto": {
        "symbols": "crypto/BTC_USDT_4h",
        "strategy_file": "strategies/crypto.py",
        # Covers 2022 bear (-75%), 2023 recovery, 2024 bull — three distinct regimes.
        "val_start": "2022-01-01",
        "val_end": "2024-12-31",
        "train_start": "2019-01-01",
        "train_end": "2021-12-31",
        "lockbox_start": "2025-01-01",
        "commission": 0.001,         # KuCoin taker (~0.1%) + thin spread padding
        "min_trades": 20,
        "max_drawdown_limit": 35.0,  # Crypto is more volatile; 35% still rejects runaway DD
        "data_fetch_start": "2019-01-01",
    },
    "stocks": {
        "symbols": "stocks/TSLA_1d,stocks/NVDA_1d,stocks/PYPL_1d",
        "strategy_file": "strategies/stocks.py",
        # Covers COVID crash, 2021 bull, 2022 bear, 2023 recovery, 2024 AI bull.
        "val_start": "2020-01-01",
        "val_end": "2024-12-31",
        "train_start": "2018-01-01",
        "train_end": "2019-12-31",
        "lockbox_start": "2025-01-01",
        "commission": 0.001,         # 10 bps round-trip; slippage + spread proxy
        "min_trades": 20,
        "max_drawdown_limit": 30.0,
        "data_fetch_start": "2015-01-01",
    },
}

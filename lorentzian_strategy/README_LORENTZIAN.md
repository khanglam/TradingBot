# Lorentzian Classification Strategy for Lumibot

This README explains how to use the Lorentzian Classification strategy with Lumibot for backtesting and live trading.

## Overview

The Lorentzian Classification strategy is a machine learning-based trading strategy that uses the Lorentzian distance metric to identify potential entry and exit points in the market. The strategy is implemented using the `advanced_ta` package, which provides the Lorentzian Classification algorithm.

## Features

- Machine learning-based trading signals using Lorentzian distance metric
- Configurable feature inputs (RSI, WT, CCI, ADX)
- Multiple filter options (volatility, regime, ADX, EMA, SMA)
- Kernel regression smoothing
- Position sizing based on portfolio percentage
- Backtesting and live trading capabilities

## Usage

### Running a Backtest

To run a backtest with the Lorentzian strategy, execute the `lorentzian_strategy.py` file:

```bash
python lorentzian_strategy.py
```

The script is configured to run in backtest mode by default. You can modify the backtest parameters in the `__main__` section of the script.

### Live Trading

To run the strategy in live trading mode, set `IS_BACKTESTING = False` in the `__main__` section and ensure your Alpaca API credentials are properly configured in your environment or in the `ALPACA_CONFIG` dictionary.

## Configuration

The strategy can be configured through the `parameters` dictionary. Here are the key parameters:

- `symbol`: The trading symbol (default: "SPY")
- `max_bars_back`: Maximum number of bars to look back for calculations (default: 2000)
- `neighbors_count`: Number of neighbors to consider in the ML algorithm (default: 8)
- `use_dynamic_exits`: Whether to use dynamic exits (default: False)
- `features`: List of feature configurations for the ML algorithm
- `use_volatility_filter`, `use_regime_filter`, `use_adx_filter`: Boolean flags for different filters
- `regime_threshold`, `adx_threshold`: Threshold values for filters
- `use_kernel_smoothing`: Whether to use kernel smoothing (default: False)
- `lookback_window`, `relative_weight`, `regression_level`, `crossover_lag`: Kernel filter parameters
- `use_ema_filter`, `ema_period`, `use_sma_filter`, `sma_period`: EMA and SMA filter settings
- `position_size`: Percentage of portfolio to allocate per trade (0.0-1.0)

## Dependencies

- lumibot
- pandas
- numpy
- pytz
- ta (Technical Analysis library)
- advanced_ta (Lorentzian Classification implementation)

## Example

Here's an example of how to customize the strategy parameters for backtesting:

```python
results, strategy = LorentzianStrategy.run_backtest(
    datasource_class=PolygonDataBacktesting,
    backtesting_start=backtesting_start,
    backtesting_end=backtesting_end,
    minutes_before_closing=0,
    benchmark_asset='SPY',
    analyze_backtest=True,
    parameters={
        "symbol": "QQQ",  # Changed from SPY to QQQ
        "max_bars_back": 1000,  # Reduced from 2000
        "neighbors_count": 10,  # Increased from 8
        "use_dynamic_exits": True,  # Changed from False
        "features": [
            {"type": "RSI", "param1": 14, "param2": 2},
            {"type": "WT", "param1": 10, "param2": 11},
            {"type": "CCI", "param1": 20, "param2": 2},
            {"type": "ADX", "param1": 20, "param2": 2},
            {"type": "RSI", "param1": 9, "param2": 2},
        ],
        "position_size": 0.5,  # Using 50% of portfolio per trade
    },
    # Other parameters...
)
```

## Notes

- The strategy requires sufficient historical data to function properly. Make sure to set `warm_up_trading_days` appropriately.
- The Lorentzian Classification algorithm is computationally intensive. Backtesting may take some time, especially with large datasets.
- Adjust the `position_size` parameter to control risk exposure.

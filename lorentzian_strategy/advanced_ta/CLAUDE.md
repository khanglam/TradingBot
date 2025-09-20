# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the `advanced_ta` directory containing a sophisticated implementation of Lorentzian Classification for financial market analysis. The system combines machine learning with technical analysis to generate trading signals and includes a comprehensive optimization framework with smart parameter discovery.

## Core Architecture

### Main Components

1. **classifier.py** - Complete Lorentzian Classification implementation
   - Uses Lorentzian distance metric instead of Euclidean for ML classification
   - Accounts for market "warping" effects from economic events
   - Features: RSI, WT (WaveTrend), CCI, ADX with normalization
   - Includes kernel regression filters and regime detection

2. **AdvancedLorentzianStrategy.py** - Lumibot-compatible strategy
   - Integrates classifier.py for live/backtesting via Lumibot framework
   - Auto-loads optimized parameters from `results_logs/best_parameters_{symbol}.json`
   - Exact position sizing: `min(cash * 0.95, cash - 1000)`
   - Long-only strategy (start_short closes longs, no actual short selling)

3. **optimize_parameters.py** - Smart optimization system
   - Intelligent parameter search with state persistence
   - Multi-strategy approach: random exploration, local search, genetic algorithm
   - Continues from previous sessions without duplicate testing
   - Saves optimization state to `optimization_state_{symbol}.pkl`

4. **test_parameters.py** - Parameter testing and validation
   - Tests parameter combinations against real market data
   - Uses exact same trading logic as AdvancedLorentzianStrategy
   - Provides performance reports and strategy validation

5. **simulate_trade.py** - Trading simulation engine
   - `AdvancedLorentzianSimulator` - Exact replica of strategy trading logic
   - `TradingSimulator` - General-purpose trading simulator
   - Ensures optimization results translate to real strategy performance

## Common Development Commands

### Environment Setup
```bash
# Set environment variables in .env file
POLYGON_API_KEY=your_key_here
SYMBOL=TSLA
BACKTESTING_START=2024-01-01
BACKTESTING_END=2024-12-01
INITIAL_CAPITAL=10000
DATA_TIMEFRAME=day
LOG_LEVEL=INFO
```

### Parameter Optimization
```bash
# Smart optimization (continues from previous sessions)
python optimize_parameters.py

# Optimization with different combinations limit
MAX_COMBINATIONS=10000 python optimize_parameters.py

# Parallel optimization
N_JOBS=4 python optimize_parameters.py

# Different log levels
LOG_LEVEL=DEBUG python optimize_parameters.py   # Detailed logs
LOG_LEVEL=INFO python optimize_parameters.py    # Progress bars (default)
LOG_LEVEL=WARN python optimize_parameters.py    # Warnings only
```

### Parameter Testing
```bash
# Test with optimized parameters
python test_parameters.py

# Test with default parameters
USE_OPTIMIZED_PARAMS=false python test_parameters.py

# Test with different symbol
SYMBOL=SPY python test_parameters.py

# Test different timeframe data
DATA_TIMEFRAME=hour python test_parameters.py
```

### Strategy Backtesting
```bash
# Run strategy with optimized parameters
python AdvancedLorentzianStrategy.py

# Use default parameters
USE_OPTIMIZED_PARAMS=false python AdvancedLorentzianStrategy.py
```

## Key Technical Concepts

### Lorentzian Classification
- Uses Lorentzian distance metric instead of Euclidean distance
- Accounts for market "warping" effects from economic events
- K-Nearest Neighbors with chronological neighbor distribution
- Features are normalized technical indicators

### Smart Optimization System
- **State Persistence**: Saves progress to resume optimization sessions
- **Multi-Strategy**: Combines random exploration, local search, and genetic algorithms
- **Adaptive Allocation**: Adjusts strategy based on progress and stagnation detection
- **Duplicate Prevention**: Never tests same parameter combination twice

### Trading Logic Consistency
- `AdvancedLorentzianSimulator` exactly replicates strategy logic
- Position sizing: `min(cash * 0.95, cash - 1000)`
- Long-only: start_long opens positions, start_short closes them
- No actual short selling (commented out in strategy)

## File Structure

```
advanced_ta/
├── classifier.py                    # Core ML classification
├── AdvancedLorentzianStrategy.py    # Lumibot strategy implementation  
├── optimize_parameters.py           # Smart parameter optimization
├── test_parameters.py              # Parameter testing and validation
├── simulate_trade.py               # Trading simulation engine
├── SMART_OPTIMIZATION.md           # Smart optimization documentation
├── logs/                           # Strategy backtest logs
│   ├── AdvancedLorentzianStrategy_*.csv
│   └── AdvancedLorentzianStrategy_*.json
└── results_logs/                   # Optimization results
    ├── best_parameters_*.json      # Best parameters for symbols
    ├── optimization_results_*.json  # Detailed optimization results
    └── optimization_state_*.pkl    # Smart optimization state
```

## Configuration and Parameters

### Environment Variables
```bash
# Core Settings
SYMBOL=TSLA                         # Trading symbol
POLYGON_API_KEY=your_key           # Data source API key
INITIAL_CAPITAL=10000              # Starting capital
USE_OPTIMIZED_PARAMS=true          # Load optimized parameters

# Data Settings  
DATA_TIMEFRAME=day                 # day/hour/minute (day recommended)
BACKTESTING_START=2024-01-01      # Backtest start date
BACKTESTING_END=2024-12-01        # Backtest end date
AGGREGATE_TO_DAILY=true           # Convert intraday to daily

# Optimization Settings
MAX_COMBINATIONS=20000             # Parameter combinations to test
USE_SMART_OPTIMIZATION=true       # Enable smart optimization
CONTINUE_FROM_BEST=true           # Continue from previous session
N_JOBS=4                          # Parallel processing cores

# Logging
LOG_LEVEL=INFO                    # DEBUG/INFO/WARN
```

### Parameter Files
- `best_parameters_{symbol}.json` - Optimized parameters for symbol
- `optimization_state_{symbol}.pkl` - Smart optimization progress
- Contains: features, ML settings, filters, kernel parameters

## Critical Implementation Notes

### Data Source Consistency
- Optimization uses Polygon API directly
- Strategy uses Lumibot's `get_historical_prices()`
- Small data differences may cause minor performance variations
- Same trading logic ensures results translate accurately

### Strategy Compatibility
- `test_parameters.py` must use `DATA_TIMEFRAME=day` for strategy compatibility
- `AdvancedLorentzianStrategy` is hardcoded for daily data
- Intraday optimization results won't translate to daily strategy

### Position Sizing Formula
All components use identical sizing: `min(cash * 0.95, cash - 1000)`
- Keeps 5% cash buffer plus $1000 minimum
- Prevents margin calls and maintains liquidity

### Signal Processing
- `isNewBuySignal` / `isNewSellSignal` - Boolean format from classifier
- `startLongTrade` / `startShortTrade` - Price format (when not NaN)
- All simulators handle both formats automatically

## Optimization Workflow

1. **First Run**: `python optimize_parameters.py`
   - Tests random parameter combinations
   - Saves best performers and optimization state
   - Creates `best_parameters_{symbol}.json`

2. **Subsequent Runs**: Automatic continuation
   - Loads previous state from `.pkl` file
   - Focuses search around best performers  
   - Uses genetic algorithms and local search

3. **Validation**: `python test_parameters.py`
   - Tests optimized parameters on validation data
   - Provides detailed performance report
   - Confirms strategy compatibility

4. **Deployment**: `python AdvancedLorentzianStrategy.py`
   - Auto-loads optimized parameters
   - Ready for live trading or backtesting

## Performance Monitoring

### Log Files
- Strategy logs: `logs/AdvancedLorentzianStrategy_*.csv`
- Optimization results: `results_logs/optimization_results_*.json`
- Plots: `results_logs/lorentzian_plot_*.jpg`

### Key Metrics
- **Optimization Score**: Combined metric balancing return, drawdown, win rate
- **Total Return**: Strategy performance vs buy & hold
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Worst peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns

## Troubleshooting

### Reset Optimization
```bash
# Clear optimization state to start fresh
rm results_logs/optimization_state_*.pkl
```

### Data Issues
```bash
# Clear cached data
python -c "from test_parameters import clear_data_cache; clear_data_cache()"
```

### Performance Issues
```bash
# Reduce combinations for faster optimization
MAX_COMBINATIONS=1000 python optimize_parameters.py

# Use fewer parallel jobs
N_JOBS=2 python optimize_parameters.py
```

### Strategy Mismatch
- Ensure `DATA_TIMEFRAME=day` for strategy compatibility
- Verify same parameter format in optimization and strategy
- Check `best_parameters_{symbol}.json` exists and is valid

## Security Considerations

- Never commit API keys to repository
- Use `.env` file for sensitive configuration
- `POLYGON_API_KEY` required for real data (no fallback to sample data)
- Validate all user inputs and parameter ranges
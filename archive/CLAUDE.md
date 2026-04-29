# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a trading bot repository that implements various trading strategies, with a focus on advanced machine learning approaches using Lorentzian Classification for financial market analysis. The project combines sentiment analysis, technical indicators, and ML-based classification to make trading decisions.

## Key Architecture Components

### Core Strategy Framework
- **Lumibot Integration**: Uses Lumibot framework for backtesting and live trading
- **Polygon Data Source**: Primary data source for market data via Polygon.io API
- **Alpaca Integration**: For live trading execution via Alpaca API

### Main Strategy Types
1. **Lorentzian Classification Strategy**: ML-based strategy using Lorentzian distance metric
2. **Traditional Strategies**: Moving average, RSI-based, and options strategies
3. **Advanced TA Strategies**: Enhanced technical analysis with optimization

### Key Directories Structure
- `lorentzian_strategy/`: Core Lorentzian Classification implementation
  - `advanced_ta/`: Enhanced version with optimization capabilities
  - `LorentzianClassificationStrategy.py`: Main strategy implementation
- `logs/`: Strategy backtest results and tearsheets
- `test_*.py`: Various test files for different components

## Common Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n trader python=3.10
conda activate trader

# Install core dependencies (always use pip3 or python -m pip)
pip3 install lumibot timedelta alpaca-trade-api==3.1.1
pip3 install torch torchvision torchaudio transformers

# Install additional ML dependencies
pip3 install scikit-learn pandas numpy matplotlib seaborn
pip3 install polygon-api-client python-dotenv

# Install all requirements
pip3 install -r requirements.txt
```

### Running Strategies
```bash
# Run basic trading bot
python tradingbot.py

# Run specific strategy backtests
python test_backtest_return.py

# Run Lorentzian strategy optimization
python lorentzian_strategy/advanced_ta/optimize_parameters.py

# Run parameter testing
python lorentzian_strategy/advanced_ta/test_parameters.py
```

### Testing
```bash
# Run individual test files
python test_backtest_return.py
python test_polygon_limits.py
python test_smart_maxbars.py
python test_optimization_fix.py
```

## Configuration and Environment

### Required Environment Variables
- `SYMBOL`: Trading symbol (default: TSLA)
- `POLYGON_API_KEY`: Polygon.io API key for data
- `ALPACA_API_KEY`: Alpaca API key for trading
- `ALPACA_SECRET_KEY`: Alpaca secret key
- `USE_INTRADAY_TIMING`: Enable intraday timing alignment
- `USE_OPTIMIZED_PARAMS`: Load optimized parameters from JSON files

### Configuration Files
- `.env`: Environment variables (not tracked in git)
- `best_parameters_{SYMBOL}.json`: Optimized parameters for specific symbols
- `requirements.txt`: Python dependencies

## Key Technical Concepts

### Lorentzian Classification
The core ML approach uses Lorentzian distance metric instead of Euclidean distance to account for market event "warping" effects. This is implemented in `classifier.py` with components:
- **Feature Engineering**: RSI, WT, CCI, ADX indicators
- **K-Nearest Neighbors**: Classification using historical patterns
- **Kernel Filters**: Smoothing and regime detection
- **Dynamic Exits**: Adaptive exit strategies

### Data Pipeline
1. **Data Ingestion**: Polygon API → pandas DataFrame
2. **Feature Calculation**: Technical indicators and ML features
3. **Signal Generation**: Lorentzian classification predictions
4. **Trade Execution**: Lumibot strategy framework
5. **Performance Tracking**: Logs and tearsheets

### Optimization Process
The optimization system (`optimize_parameters.py`) uses:
- **Parameter Grid Search**: Testing multiple parameter combinations
- **Backtesting Validation**: Historical performance validation
- **Result Storage**: JSON files with best parameters
- **Timing Alignment**: Intraday timing to prevent look-ahead bias

## Critical Implementation Details

### Data Timing Alignment
- **Issue**: Optimization used daily close data but strategy trades at 10:30 AM
- **Solution**: `USE_INTRADAY_TIMING=true` downloads hourly data filtered to 10:30 AM
- **Impact**: Eliminates look-ahead bias that caused 150%+ performance discrepancies

### Parameter Consistency
- Strategy uses identical `classifier.py` implementation as optimization
- Parameters loaded from `best_parameters_{SYMBOL}.json` files
- Standardized data structures across all components

### Performance Monitoring
- All backtests generate HTML tearsheets in `logs/` directory
- Strategy performance tracked with detailed CSV exports
- Optimization results stored in `results_logs/` directory

## Important Notes

### SSL Certificate Issues
If encountering SSL errors with Alpaca API, download and install:
- https://letsencrypt.org/certs/lets-encrypt-r3.pem (rename to .cer)
- https://letsencrypt.org/certs/isrg-root-x1-cross-signed.pem (rename to .cer)

### Data Source Requirements
- Polygon.io API key required (free tier available)
- Alpaca account for live trading
- Historical data cached for optimization performance

### Browser Suppression
The codebase includes browser suppression for headless operation:
```python
# Monkey patch to prevent browser windows during backtesting
webbrowser.open = no_op_open
```

## Development Workflow

1. **Strategy Development**: Implement in `lorentzian_strategy/` directory
2. **Parameter Optimization**: Use `optimize_parameters.py` to find optimal settings
3. **Backtesting**: Validate with `test_parameters.py` 
4. **Performance Analysis**: Review tearsheets in `logs/` directory
5. **Live Trading**: Deploy with Alpaca integration

## Common Pitfalls to Avoid

- **Look-ahead Bias**: Ensure timing alignment between optimization and strategy
- **Data Source Consistency**: Use same data source for optimization and backtesting
- **Parameter Overfitting**: Validate on out-of-sample data
- **Environment Setup**: Always use pip3/python -m pip for conda environments
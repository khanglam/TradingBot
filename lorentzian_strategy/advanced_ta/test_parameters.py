"""
Advanced TA Demo Script
======================

This script demonstrates how to use the lorentzian_strategy/advanced_ta package
for machine learning-based market classification.

Run this script with: python test_parameters.py

For different log levels, set LOG_LEVEL in your .env file or run:
LOG_LEVEL=DEBUG python test_parameters.py  # Shows detailed logs including individual trades
LOG_LEVEL=INFO python test_parameters.py   # Shows progress and summaries (default)
LOG_LEVEL=WARN python test_parameters.py   # Shows only warnings and errors
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from polygon import RESTClient

# Load environment variables
load_dotenv()

# Logging level control
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

def is_debug():
    """Returns True if DEBUG level logging is enabled"""
    return LOG_LEVEL == 'DEBUG'

def is_info():
    """Returns True if INFO level logging is enabled (includes DEBUG)"""
    return LOG_LEVEL in ['DEBUG', 'INFO']

def is_warn():
    """Returns True if WARN level logging is enabled (includes all levels)"""
    return LOG_LEVEL in ['DEBUG', 'INFO', 'WARN']

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Global data cache to avoid repeated API calls
_data_cache = {}

def get_next_file_number(directory: str, base_filename: str, extension: str) -> int:
    """
    Get the next available number for incremental file naming
    
    Args:
        directory: Directory to search for existing files
        base_filename: Base filename pattern (e.g., "lorentzian_plot_TSLA")
        extension: File extension (e.g., "jpg")
        
    Returns:
        Next available number (1 if no files exist)
    """
    if not os.path.exists(directory):
        return 1
    
    import glob
    pattern = os.path.join(directory, f"{base_filename}_*.{extension}")
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return 1
    
    # Extract numbers from existing files
    numbers = []
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        # Remove base filename and extension, extract number
        try:
            # Pattern: base_filename_NUMBER.extension
            number_part = filename.replace(f"{base_filename}_", "").replace(f".{extension}", "")
            if number_part.isdigit():
                numbers.append(int(number_part))
        except:
            continue
    
    if not numbers:
        return 1
    
    return max(numbers) + 1

def generate_incremental_filename(directory: str, base_filename: str, extension: str) -> str:
    """
    Generate an incremental filename (e.g., lorentzian_plot_TSLA_1.jpg)
    
    Args:
        directory: Directory where file will be saved
        base_filename: Base filename pattern
        extension: File extension (without dot)
        
    Returns:
        Full path with incremental number
    """
    next_number = get_next_file_number(directory, base_filename, extension)
    filename = f"{base_filename}_{next_number}.{extension}"
    return os.path.join(directory, filename)

def clear_data_cache():
    """Clear the data cache"""
    global _data_cache
    _data_cache = {}
    print("🗑️  Data cache cleared")

def get_cache_info():
    """Get information about cached data"""
    if not _data_cache:
        print("📋 No data cached")
        return
    
    print(f"📋 Cached datasets: {len(_data_cache)}")
    for key, df in _data_cache.items():
        print(f"   • {key}: {len(df)} rows")

try:
    from classifier import (
        LorentzianClassification, 
        Feature, 
        Settings, 
        FilterSettings, 
        KernelFilter,
        Direction
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the advanced_ta directory")
    sys.exit(1)

def download_real_data(symbol=None, start_date=None, end_date=None, timeframe='day'):
    """
    Download real market data from Polygon API with improved error handling and caching
    
    Args:
        symbol (str): Stock symbol to download
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        timeframe (str): 'day' for daily data, 'hour' for hourly data, 'minute' for minute data
    
    Returns:
        pd.DataFrame: OHLCV data with DatetimeIndex
    """
    # Check cache first to avoid repeated API calls
    cache_key = f"{symbol}_{start_date}_{end_date}_{timeframe}"
    if cache_key in _data_cache:
        print(f"📋 Using cached {timeframe} data for {symbol} ({start_date} to {end_date})")
        return _data_cache[cache_key].copy()
    
    print(f"📥 Downloading real {timeframe} data for {symbol} from Polygon...")
    
    # Get API key from environment
    polygon_api_key = os.getenv('POLYGON_API_KEY')
    if not polygon_api_key:
        print("❌ POLYGON_API_KEY not found in environment variables")
        print("💡 Set your API key: export POLYGON_API_KEY='your_key_here'")
        raise ValueError("POLYGON_API_KEY is required for real data. Sample data is not useful for trading.")
    
    try:
        # Retry logic with exponential backoff
        max_retries = 3
        base_delay = 2.0
        aggs = []
        
        for attempt in range(max_retries):
            try:
                # Initialize Polygon client
                polygon_client = RESTClient(polygon_api_key)
                
                if is_debug():
                    print(f"📊 Fetching {timeframe} data from {start_date} to {end_date} (attempt {attempt + 1}/{max_retries})")
                
                # Add delay to avoid rate limiting
                if attempt > 0:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    if is_debug():
                        print(f"⏱️  Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                else:
                    time.sleep(1.0)  # Initial delay
                
                # Get aggregates from Polygon
                aggs = []
                
                if timeframe == 'minute':
                    # For minute data, we need to handle potential large datasets
                    if is_debug():
                        print(f"⚠️  Warning: Minute data downloads can be very large and slow!")
                        print(f"   Consider using shorter date ranges for minute data")
                        
                        # Calculate expected data points
                        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                        days_diff = (end_dt - start_dt).days
                        estimated_points = days_diff * 390  # ~390 trading minutes per day
                        
                        print(f"   Estimated data points: {estimated_points:,}")
                        if estimated_points > 50000:
                            print(f"   🚨 This is a LOT of data! Consider shorter date range.")
                    
                    for agg in polygon_client.get_aggs(
                        ticker=symbol,
                        multiplier=1,
                        timespan="minute",
                        from_=start_date,
                        to=end_date,
                        limit=50000  # Polygon limit
                    ):
                        aggs.append(agg)
                        
                elif timeframe == 'hour':
                    # For hourly data
                    if is_debug():
                        print(f"📊 Downloading hourly data...")
                        
                        # Calculate expected data points
                        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                        days_diff = (end_dt - start_dt).days
                        estimated_points = days_diff * 6.5  # ~6.5 trading hours per day
                        
                        print(f"   Estimated data points: {estimated_points:,}")
                        if estimated_points > 10000:
                            print(f"   ⚠️  Large dataset - may take some time to download.")
                    
                    for agg in polygon_client.get_aggs(
                        ticker=symbol,
                        multiplier=1,
                        timespan="hour",
                        from_=start_date,
                        to=end_date,
                        limit=50000  # Polygon limit
                    ):
                        aggs.append(agg)
                        
                else:
                    # Daily data (original logic)
                    for agg in polygon_client.get_aggs(
                        ticker=symbol,
                        multiplier=1,
                        timespan="day",
                        from_=start_date,
                        to=end_date,
                        limit=5000
                    ):
                        aggs.append(agg)
                
                # If we get here, the API call succeeded
                break
                
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    print(f"⚠️  Rate limited (attempt {attempt + 1}/{max_retries}), retrying...")
                    continue
                else:
                    # Re-raise the exception if it's the last attempt or not a rate limit error
                    raise e
        
        # Process the downloaded data
        if not aggs:
            raise ValueError(f"No {timeframe} data found for {symbol} from Polygon. Check symbol and date range.")
        
        # Convert to DataFrame
        data_list = []
        for agg in aggs:
            data_list.append({
                'date': datetime.fromtimestamp(agg.timestamp / 1000),  # Keep as datetime for DatetimeIndex
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume
            })
        
        df = pd.DataFrame(data_list)
        df = df.sort_values('date').reset_index(drop=True)
        df.set_index('date', inplace=True)
        
        # Ensure index is DatetimeIndex (required for mplfinance)
        df.index = pd.to_datetime(df.index)
        
        # Display download results
        if timeframe == 'minute':
            data_type = "minutes"
        elif timeframe == 'hour':
            data_type = "hours"
        else:
            data_type = "days"
            
        print(f"✅ Downloaded {len(df)} {data_type} of real {timeframe} data from Polygon")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        
        if timeframe in ['minute', 'hour'] and is_info():
            # Additional stats for intraday data
            unique_dates = pd.Series(df.index.date).unique()
            trading_days = len(unique_dates)
            
            if timeframe == 'minute':
                avg_per_day = len(df) / trading_days if trading_days > 0 else 0
                print(f"   Trading days: {trading_days}")
                print(f"   Avg minutes per day: {avg_per_day:.1f}")
            else:  # hour
                avg_per_day = len(df) / trading_days if trading_days > 0 else 0
                print(f"   Trading days: {trading_days}")
                print(f"   Avg hours per day: {avg_per_day:.1f}")
            
            # Estimate file size
            estimated_size_mb = len(df) * 6 * 8 / (1024 * 1024)  # 6 columns, 8 bytes each
            print(f"   Estimated cache size: {estimated_size_mb:.1f} MB")
        
        # Cache the data to avoid repeated API calls
        _data_cache[cache_key] = df.copy()
        if is_info():
            print(f"💾 {timeframe.title()} data cached for future use")
        
        return df
        
    except Exception as e:
        print(f"❌ Failed to download {timeframe} data from Polygon: {e}")
        print(f"💡 Possible solutions:")
        print(f"   1. Check your POLYGON_API_KEY is valid")
        print(f"   2. Verify the symbol '{symbol}' exists")
        print(f"   3. Check date range: {start_date} to {end_date}")
        print(f"   4. Wait a few minutes if hitting rate limits")
        print(f"   5. Consider upgrading your Polygon plan for higher rate limits")
        if timeframe == 'minute':
            print(f"   6. For minute data, try a shorter date range (e.g., 1-7 days)")
        elif timeframe == 'hour':
            print(f"   6. For hourly data, try a shorter date range (e.g., 1-30 days)")
        raise Exception(f"Real {timeframe} data download failed: {e}. Sample data is not acceptable for trading.")

def aggregate_minute_to_daily(df_minute):
    """
    Aggregate minute data to daily OHLCV data
    
    Args:
        df_minute (pd.DataFrame): Minute OHLCV data with DatetimeIndex
    
    Returns:
        pd.DataFrame: Daily OHLCV data
    """
    if is_info():
        print(f"📊 Aggregating {len(df_minute)} minute bars to daily data...")
    
    # Group by date and aggregate
    daily_data = df_minute.groupby(df_minute.index.date).agg({
        'open': 'first',    # First price of the day
        'high': 'max',      # Highest price of the day
        'low': 'min',       # Lowest price of the day
        'close': 'last',    # Last price of the day
        'volume': 'sum'     # Total volume for the day
    })
    
    # Convert date index back to datetime
    daily_data.index = pd.to_datetime(daily_data.index)
    daily_data.index.name = 'date'
    
    if is_info():
        print(f"✅ Aggregated to {len(daily_data)} daily bars")
        print(f"   Date range: {daily_data.index[0].date()} to {daily_data.index[-1].date()}")
    
    return daily_data

def aggregate_hour_to_daily(df_hour):
    """
    Aggregate hourly data to daily OHLCV data
    
    Args:
        df_hour (pd.DataFrame): Hourly OHLCV data with DatetimeIndex
    
    Returns:
        pd.DataFrame: Daily OHLCV data
    """
    if is_info():
        print(f"📊 Aggregating {len(df_hour)} hourly bars to daily data...")
    
    # Group by date and aggregate
    daily_data = df_hour.groupby(df_hour.index.date).agg({
        'open': 'first',    # First price of the day
        'high': 'max',      # Highest price of the day
        'low': 'min',       # Lowest price of the day
        'close': 'last',    # Last price of the day
        'volume': 'sum'     # Total volume for the day
    })
    
    # Convert date index back to datetime
    daily_data.index = pd.to_datetime(daily_data.index)
    daily_data.index.name = 'date'
    
    if is_info():
        print(f"✅ Aggregated to {len(daily_data)} daily bars")
        print(f"   Date range: {daily_data.index[0].date()} to {daily_data.index[-1].date()}")
    
    return daily_data

def aggregate_intraday_to_daily(df_intraday, source_timeframe):
    """
    Generic function to aggregate intraday data (minute or hour) to daily OHLCV data
    
    Args:
        df_intraday (pd.DataFrame): Intraday OHLCV data with DatetimeIndex
        source_timeframe (str): 'minute' or 'hour'
    
    Returns:
        pd.DataFrame: Daily OHLCV data
    """
    if source_timeframe == 'minute':
        return aggregate_minute_to_daily(df_intraday)
    elif source_timeframe == 'hour':
        return aggregate_hour_to_daily(df_intraday)
    else:
        raise ValueError(f"Unsupported timeframe for aggregation: {source_timeframe}")

def generate_sample_data_fallback(symbol=None, days=300):
    """Generate sample OHLCV data as fallback"""
    print(f"📊 Generating {days} days of sample data for {symbol}...")
    
    # Set random seed for reproducibility (only used for sample data fallback)
    np.random.seed(42)
    
    # Generate dates (business days only)
    start_date = datetime.now() - timedelta(days=int(days * 1.4))
    dates = pd.date_range(start=start_date, periods=days, freq='B')
    
    # Generate realistic price data using geometric Brownian motion
    initial_price = 200.0
    returns = np.random.normal(0.0005, 0.02, days)
    
    prices = [initial_price]
    for i in range(1, days):
        price = prices[-1] * (1 + returns[i])
        prices.append(price)
    
    # Generate OHLC from close prices
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        daily_range = abs(np.random.normal(0, 0.015)) * close
        high = close + np.random.uniform(0, daily_range)
        low = close - np.random.uniform(0, daily_range)
        open_price = low + np.random.uniform(0, high - low)
        
        # Ensure OHLC logic is maintained
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        volume = int(np.random.uniform(1000000, 10000000))
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    # Ensure index is DatetimeIndex (required for mplfinance)
    df.index = pd.to_datetime(df.index)
    
    print(f"✅ Generated sample data: {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    return df

def load_optimized_parameters(symbol):
    """Load optimized parameters from JSON file if available"""
    # Look for the file in the same directory as this script
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_logs")
    best_params_file = os.path.join(script_dir, f"best_parameters_{symbol}.json")
    
    if os.path.exists(best_params_file):
        try:
            with open(best_params_file, 'r') as f:
                data = json.load(f)
            
            print(f"✅ Found optimized parameters for {symbol}")
            print(f"   Optimization date: {data['optimization_info']['optimization_date']}")
            print(f"   Optimization score: {data['optimization_info']['optimization_score']:.3f}")
            print(f"   Expected return: {data['optimization_info']['total_return']:+.2f}%")
            print(f"   Expected win rate: {data['optimization_info']['win_rate']:.1f}%")
            
            return data['best_parameters']
            
        except Exception as e:
            print(f"❌ Error loading optimized parameters: {e}")
            return None
    else:
        print(f"ℹ️  No optimized parameters found for {symbol}")
        print(f"   Run 'python optimize_parameters.py' to generate optimized parameters")
        return None

def create_features_from_params(params):
    """Create Feature objects from parameter data"""
    if params is None:
        # Default features if no optimized parameters
        return [
            Feature("RSI", 14, 1),
            Feature("WT", 9, 10),
            Feature("CCI", 14, 1),
        ]
    
    features = []
    for feature_data in params['features']:
        features.append(Feature(
            feature_data['type'],
            feature_data['param1'],
            feature_data['param2']
        ))
    
    return features

def create_settings_from_params(params, df_source):
    """Create Settings object from parameter data"""
    if params is None:
        # Default settings if no optimized parameters
        return Settings(
            source=df_source,
            neighborsCount=6,
            maxBarsBack=1500,
            useDynamicExits=True,
            useEmaFilter=False,
            emaPeriod=200,
            useSmaFilter=False,
            smaPeriod=200
        )
    
    return Settings(
        source=df_source,
        neighborsCount=params['neighborsCount'],
        maxBarsBack=params['maxBarsBack'],
        useDynamicExits=params['useDynamicExits'],
        useEmaFilter=params['useEmaFilter'],
        emaPeriod=params['emaPeriod'],
        useSmaFilter=params['useSmaFilter'],
        smaPeriod=params['smaPeriod']
    )

def create_filter_settings_from_params(params):
    """Create FilterSettings object from parameter data"""
    if params is None:
        # Default filter settings if no optimized parameters
        kernel_filter = KernelFilter(
            useKernelSmoothing=False,
            lookbackWindow=8,
            relativeWeight=8.0,
            regressionLevel=25,
            crossoverLag=2
        )
        
        return FilterSettings(
            useVolatilityFilter=False,
            useRegimeFilter=False,
            useAdxFilter=False,
            regimeThreshold=0.0,
            adxThreshold=20,
            kernelFilter=kernel_filter
        )
    
    # Create kernel filter from parameters
    kernel_filter = KernelFilter(
        useKernelSmoothing=params['kernel_filter']['useKernelSmoothing'],
        lookbackWindow=params['kernel_filter']['lookbackWindow'],
        relativeWeight=params['kernel_filter']['relativeWeight'],
        regressionLevel=params['kernel_filter']['regressionLevel'],
        crossoverLag=params['kernel_filter']['crossoverLag']
    )
    
    return FilterSettings(
        useVolatilityFilter=params['filter_settings']['useVolatilityFilter'],
        useRegimeFilter=params['filter_settings']['useRegimeFilter'],
        useAdxFilter=params['filter_settings']['useAdxFilter'],
        regimeThreshold=params['filter_settings']['regimeThreshold'],
        adxThreshold=params['filter_settings']['adxThreshold'],
        kernelFilter=kernel_filter
    )

def simulate_trading_strategy(df, features, settings, filter_settings, symbol, initial_capital=10000):
    """
    Simulate trading strategy using EXACT AdvancedLorentzianStrategy logic
    
    This function now uses the AdvancedLorentzianSimulator which exactly replicates
    the trading logic from AdvancedLorentzianStrategy.py to ensure optimization
    results translate perfectly to real strategy performance.
    
    Key features that match AdvancedLorentzianStrategy exactly:
    - Position sizing: min(cash * 0.95, cash - 1000)
    - Trading logic: start_long opens long, start_short closes long (no short selling)
    - Signal processing: Uses latest signals from classifier
    - Cash management: Same buffer and sizing logic
    """
    from simulate_trade import run_advanced_lorentzian_simulation
    
    # Use the EXACT AdvancedLorentzianStrategy simulation
    results = run_advanced_lorentzian_simulation(
        df=df,
        features=features,
        settings=settings,
        filter_settings=filter_settings,
        initial_capital=initial_capital
    )
    
    # Extract metrics from results
    metrics = results['metrics']
    
    # Add symbol to metrics (the new function doesn't include it)
    metrics['symbol'] = symbol
    
    # Convert Trade dataclass objects to dictionaries for compatibility
    if 'all_trades' in metrics:
        trade_dicts = []
        for trade in metrics['all_trades']:
            trade_dicts.append({
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'side': trade.side,
                'return_pct': trade.return_pct,
                'return_dollars': trade.return_dollars,
                'days_held': trade.days_held,
                'reason': trade.reason
            })
        metrics['all_trades'] = trade_dicts
    
    return metrics

def display_performance_report(metrics):
    """Display a beautifully formatted performance report"""
    if not metrics:
        print("\n❌ No completed trades found - cannot calculate performance metrics")
        return
    
    print("\n" + "="*80)
    print("🏆 TRADING PERFORMANCE REPORT")
    print("="*80)
    
    # Header info
    print(f"📊 Symbol: {metrics['symbol']}")
    print(f"📅 Period: {metrics['start_date'].strftime('%Y-%m-%d')} to {metrics['end_date'].strftime('%Y-%m-%d')} ({metrics['total_days']} days)")
    print(f"🔄 Total Trades: {metrics['total_trades']} (Long: {metrics['long_trades']}, Short: {metrics['short_trades']})")
    
    print("\n" + "-"*50)
    print("💰 PORTFOLIO VALUE")
    print("-"*50)
    
    print(f"💵 Initial Capital:     ${metrics['initial_capital']:,.2f}")
    
    final_color = "🟢" if metrics['final_portfolio_value'] > metrics['initial_capital'] else "🔴"
    print(f"{final_color} Final Portfolio:    ${metrics['final_portfolio_value']:,.2f}")
    
    dollar_return_color = "🟢" if metrics['total_dollar_return'] > 0 else "🔴" if metrics['total_dollar_return'] < 0 else "⚪"
    print(f"{dollar_return_color} Total P&L:          ${metrics['total_dollar_return']:+,.2f}")
    
    print(f"💸 Avg P&L per Trade:   ${metrics['avg_dollar_return_per_trade']:+,.2f}")
    
    print("\n" + "-"*50)
    print("📊 PERCENTAGE RETURNS")
    print("-"*50)
    
    # Color coding for returns
    total_return_color = "🟢" if metrics['total_return'] > 0 else "🔴" if metrics['total_return'] < 0 else "⚪"
    buy_hold_color = "🟢" if metrics['buy_hold_return'] > 0 else "🔴" if metrics['buy_hold_return'] < 0 else "⚪"
    
    print(f"{total_return_color} Strategy Return:     {metrics['total_return']:+7.2f}%")
    print(f"📈 Avg Return/Trade:   {metrics['avg_return_per_trade']:+7.2f}%")
    print(f"{buy_hold_color} Buy & Hold Return:  {metrics['buy_hold_return']:+7.2f}%")
    
    outperformance = metrics['total_return'] - metrics['buy_hold_return']
    outperf_color = "🟢" if outperformance > 0 else "🔴" if outperformance < 0 else "⚪"
    print(f"{outperf_color} Strategy vs B&H:    {outperformance:+7.2f}%")
    
    print("\n" + "-"*50)
    print("💰 BUY & HOLD COMPARISON")
    print("-"*50)
    
    bh_final_color = "🟢" if metrics['buy_hold_final_value'] > metrics['initial_capital'] else "🔴"
    print(f"{bh_final_color} B&H Final Value:    ${metrics['buy_hold_final_value']:,.2f}")
    
    bh_dollar_color = "🟢" if metrics['buy_hold_dollar_return'] > 0 else "🔴" if metrics['buy_hold_dollar_return'] < 0 else "⚪"
    print(f"{bh_dollar_color} B&H Total P&L:      ${metrics['buy_hold_dollar_return']:+,.2f}")
    
    dollar_outperf = metrics['total_dollar_return'] - metrics['buy_hold_dollar_return']
    dollar_outperf_color = "🟢" if dollar_outperf > 0 else "🔴" if dollar_outperf < 0 else "⚪"
    print(f"{dollar_outperf_color} Strategy vs B&H:    ${dollar_outperf:+,.2f}")
    
    print("\n" + "-"*50)
    print("🎯 WIN/LOSS METRICS")
    print("-"*50)
    
    win_rate_color = "🟢" if metrics['win_rate'] >= 50 else "🟡" if metrics['win_rate'] >= 40 else "🔴"
    print(f"{win_rate_color} Win Rate:           {metrics['win_rate']:7.1f}%")
    print(f"🏆 Average Win:        {metrics['avg_win']:+7.2f}%")
    print(f"💸 Average Loss:       {metrics['avg_loss']:+7.2f}%")
    
    # Win/Loss ratio - fix division by zero
    win_loss_ratio = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else float('inf')
    ratio_color = "🟢" if win_loss_ratio > 1.5 else "🟡" if win_loss_ratio > 1.0 else "🔴"
    print(f"{ratio_color} Win/Loss Ratio:     {win_loss_ratio:7.2f}")
    
    # Profit factor
    pf_color = "🟢" if metrics['profit_factor'] > 1.5 else "🟡" if metrics['profit_factor'] > 1.0 else "🔴"
    pf_display = f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] != float('inf') else "∞"
    print(f"{pf_color} Profit Factor:      {pf_display:>7}")
    
    print("\n" + "-"*50)
    print("⚡ RISK METRICS")
    print("-"*50)
    
    # Sharpe ratio
    sharpe_color = "🟢" if metrics['sharpe_ratio'] > 1.0 else "🟡" if metrics['sharpe_ratio'] > 0.5 else "🔴"
    print(f"{sharpe_color} Sharpe Ratio:       {metrics['sharpe_ratio']:7.2f}")
    
    # Max drawdown
    dd_color = "🟢" if metrics['max_drawdown'] > -5 else "🟡" if metrics['max_drawdown'] > -15 else "🔴"
    print(f"{dd_color} Max Drawdown:       {metrics['max_drawdown']:7.2f}%")
    print(f"📊 Return Volatility:   {metrics['std_return']:7.2f}%")
    
    print("\n" + "-"*50)
    print("⏱️  TRADING FREQUENCY")
    print("-"*50)
    
    print(f"📅 Trades per Month:    {metrics['trades_per_month']:7.1f}")
    print(f"⏳ Avg Holding Period:  {metrics['avg_holding_days']:7.1f} days")
    
    print("\n" + "="*80)
    
    # Performance summary
    if metrics['total_return'] > metrics['buy_hold_return'] and metrics['win_rate'] > 50:
        print("🎉 EXCELLENT: Strategy outperformed buy & hold with good win rate!")
    elif metrics['total_return'] > metrics['buy_hold_return']:
        print("✅ GOOD: Strategy outperformed buy & hold")
    elif metrics['win_rate'] > 50:
        print("👍 DECENT: Good win rate but underperformed buy & hold")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Consider adjusting strategy parameters")
    
    print("="*80)

def validate_configuration_for_strategy_match():
    """
    Validate configuration to ensure optimization results match AdvancedLorentzianStrategy
    
    This function ensures that test_parameters.py uses the same configuration as
    AdvancedLorentzianStrategy.py for meaningful optimization results.
    """
    timeframe = os.getenv('DATA_TIMEFRAME', 'day').lower()
    
    print("🔍 STRATEGY COMPATIBILITY VALIDATION")
    print("="*60)
    
    # Check timeframe compatibility
    if timeframe != 'day':
        print(f"⚠️  WARNING: Data timeframe mismatch!")
        print(f"   test_parameters.py timeframe: {timeframe}")
        print(f"   AdvancedLorentzianStrategy timeframe: day (hardcoded)")
        print(f"   ")
        print(f"   🚨 CRITICAL: This mismatch means optimization results")
        print(f"      will NOT translate to real strategy performance!")
        print(f"   ")
        print(f"   🔧 SOLUTION: Set DATA_TIMEFRAME=day in your .env file")
        print(f"      or run: DATA_TIMEFRAME=day python test_parameters.py")
        print(f"   ")
        
        response = input(f"   Force daily data for strategy compatibility? (Y/n): ").strip().lower()
        if response in ['', 'y', 'yes']:
            print(f"   ✅ Switching to daily data for strategy compatibility")
            return 'day'
        else:
            print(f"   ⚠️  Continuing with {timeframe} data - results may not translate to strategy")
            return timeframe
    else:
        print(f"✅ Timeframe: {timeframe} (matches AdvancedLorentzianStrategy)")
    
    # Data source warning
    print(f"⚠️  Data Source Difference:")
    print(f"   test_parameters.py: Polygon API direct")
    print(f"   AdvancedLorentzianStrategy: Lumibot get_historical_prices()")
    print(f"   ")
    print(f"   📊 Note: Small differences in data may cause minor performance variations")
    print(f"      but the trading logic is now EXACTLY matched.")
    
    # Trading logic confirmation
    print(f"✅ Trading Logic: EXACT match with AdvancedLorentzianStrategy")
    print(f"   • Position sizing: min(cash * 0.95, cash - 1000)")
    print(f"   • start_long: Opens long positions")  
    print(f"   • start_short: Closes long positions (no short selling)")
    print(f"   • Signal processing: Latest signals from classifier")
    
    print("="*60)
    print()
    
    return timeframe

def main():
    """Main demo function"""
    print("🎯 Starting Advanced TA Lorentzian Classification Demo")
    print(f"   Working directory: {os.getcwd()}")
    print()
    
    # Validate configuration for strategy compatibility
    validated_timeframe = validate_configuration_for_strategy_match()
    
    # Download real market data
    symbol = os.getenv('SYMBOL', 'TSLA')
    start_date = os.getenv('BACKTESTING_START', '2024-01-31')
    end_date = os.getenv('BACKTESTING_END', '2024-12-31')
    initial_capital = float(os.getenv('INITIAL_CAPITAL', '10000'))
    use_optimized_params = os.getenv('USE_OPTIMIZED_PARAMS', 'true').lower() == 'true'
    timeframe = validated_timeframe  # Use validated timeframe
    
    if is_info():
        print(f"📊 Configuration:")
        print(f"   Symbol: {symbol}")
        print(f"   Date range: {start_date} to {end_date}")
        print(f"   Data timeframe: {timeframe}")
        print(f"   Initial capital: ${initial_capital:,.2f}")
        print(f"   Use optimized parameters: {use_optimized_params}")
    else:
        print(f"Testing {symbol} ({start_date} to {end_date}) with ${initial_capital:,.0f} capital")
    
    # Validate timeframe
    if timeframe not in ['day', 'hour', 'minute']:
        print(f"⚠️  Invalid timeframe '{timeframe}', defaulting to 'day'")
        timeframe = 'day'
    
    # Warning for intraday data
    if timeframe in ['minute', 'hour']:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days_diff = (end_dt - start_dt).days
        
        if timeframe == 'minute' and days_diff > 30:
            print(f"⚠️  WARNING: Requesting {days_diff} days of minute data!")
            print(f"   This could be {days_diff * 390:,} data points and take a very long time.")
            print(f"   Consider using a shorter date range for minute data.")
            response = input(f"   Continue? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print(f"   Exiting...")
                return
        elif timeframe == 'hour' and days_diff > 365:
            print(f"⚠️  WARNING: Requesting {days_diff} days of hourly data!")
            print(f"   This could be {days_diff * 6.5:,.0f} data points and take some time.")
            print(f"   Consider using a shorter date range for better performance.")
            response = input(f"   Continue? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print(f"   Exiting...")
                return
    
    df = download_real_data(symbol=symbol, start_date=start_date, end_date=end_date, timeframe=timeframe)
    
    # Handle intraday data aggregation
    if timeframe in ['minute', 'hour']:
        print(f"\n📊 Note: Lorentzian Classification works best with daily data")
        aggregate_choice = os.getenv('AGGREGATE_TO_DAILY', 'true').lower()
        
        if aggregate_choice == 'true':
            print(f"🔄 Aggregating {timeframe} data to daily bars...")
            df_daily = aggregate_intraday_to_daily(df, timeframe)
            
            # Keep both datasets available
            df_intraday = df.copy()  # Original intraday data
            df = df_daily            # Use daily for classification
            
            print(f"✅ Using daily aggregated data for classification")
            print(f"   Original {timeframe} data: {len(df_intraday)} bars")
            print(f"   Aggregated daily data: {len(df)} bars")
        else:
            print(f"⚠️  Using raw {timeframe} data for classification")
            print(f"   This may produce many signals and slower performance")
    
    # Load optimized parameters if available and requested
    optimized_params = None
    if use_optimized_params:
        if is_info():
            print(f"\n🔍 Checking for optimized parameters...")
        optimized_params = load_optimized_parameters(symbol)
    
    # Create features (optimized or default)
    features = create_features_from_params(optimized_params)
    
    if is_info():
        print(f"\n📈 Features for classification:")
        for i, feature in enumerate(features, 1):
            print(f"   {i}. {feature.type}({feature.param1}, {feature.param2})")
        
        if optimized_params:
            print(f"   🎯 Using optimized parameters!")
    else:
        param_source = "optimized" if optimized_params else "default"
        print(f"Using {param_source} parameters with {len(features)} features")
    
    # Run classification
    if is_info():
        print(f"\n🧠 Running Lorentzian Classification...")
    else:
        print(f"Running classification...")
    
    try:
        from classifier import LorentzianClassification, Settings, FilterSettings, KernelFilter
        
        # Convert to the format expected by advanced_ta (lowercase columns)
        df_for_classification = df.copy()
        df_for_classification.columns = df_for_classification.columns.str.lower()
        
        # Create settings and filters (optimized or default)
        settings = create_settings_from_params(optimized_params, df_for_classification['close'])
        filter_settings = create_filter_settings_from_params(optimized_params)
        
        if is_info():
            print(f"\n⚙️  Classification Settings:")
            print(f"   Neighbors: {settings.neighborsCount}")
            print(f"   Max bars back: {settings.maxBarsBack}")
            print(f"   Dynamic exits: {settings.useDynamicExits}")
            
            if optimized_params:
                print(f"   🎯 Using optimized filter settings")
            else:
                print(f"   Filters: All disabled for more signals")
        
        lc = LorentzianClassification(df_for_classification, features, settings, filter_settings)
        
        if is_info():
            print(f"✅ Classification completed successfully!")
        else:
            print(f"Classification completed")
        
        # Create results directory and plot file path
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_logs")
        os.makedirs(results_dir, exist_ok=True)
        
        plot_file = generate_incremental_filename(results_dir, f"lorentzian_plot_{symbol}", "jpg")
        
        if is_info():
            print(f"📊 Generating plot...")
        try:
            lc.plot(plot_file)
            if is_info():
                print(f"✅ Plot saved to: {plot_file}")
        except ImportError as plot_error:
            if "mplfinance" in str(plot_error) and is_info():
                print(f"⚠️  Plot generation skipped - missing dependency")
                print(f"   Install with: pip install mplfinance")
            elif is_info():
                raise plot_error
        except Exception as plot_error:
            if is_info():
                print(f"⚠️  Plot generation failed: {str(plot_error)}")
                print(f"   Continuing with trading simulation...")
        
        # Simulate trading strategy (mimics AdvancedLorentzianStrategy logic)
        if is_info():
            print(f"\n📊 Simulating Trading Strategy...")
        else:
            print(f"Running trading simulation...")
        metrics = simulate_trading_strategy(df_for_classification, features, settings, filter_settings, symbol, initial_capital)
        
        if metrics:
            if is_info():
                display_performance_report(metrics)
            else:
                # Simple summary for non-verbose mode
                print(f"Results: {metrics['total_return']:+.1f}% return, {metrics['win_rate']:.0f}% win rate, {metrics['total_trades']} trades")
                print(f"Final value: ${metrics['final_portfolio_value']:,.0f} (vs B&H: {metrics['buy_hold_return']:+.1f}%)")
        else:
            print("❌ No trades were generated - check strategy parameters")
        
    except Exception as e:
        print(f"❌ Classification failed: {str(e)}")
        if "mplfinance" in str(e):
            print(f"💡 Install missing dependency: pip install mplfinance")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
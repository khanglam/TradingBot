"""
Advanced TA Demo Script
======================

This script demonstrates how to use the lorentzian_strategy/advanced_ta package
for machine learning-based market classification.

Run this script with: python run_advanced_ta_demo.py
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from polygon import RESTClient

# Load environment variables
load_dotenv()

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from lorentzian_classification import (
        LorentzianClassification, 
        Feature, 
        Settings, 
        FilterSettings, 
        KernelFilter,
        Direction
    )
    print("✅ Successfully imported LorentzianClassification components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the advanced_ta directory")
    sys.exit(1)

def download_real_data(symbol='SPY', start_date='2023-01-01', end_date='2024-12-31'):
    """Download real market data from Polygon API"""
    print(f"📥 Downloading real data for {symbol} from Polygon...")
    
    # Get API key from environment
    polygon_api_key = os.getenv('POLYGON_API_KEY')
    if not polygon_api_key:
        print("❌ POLYGON_API_KEY not found in environment variables")
        print("💡 Set your API key: export POLYGON_API_KEY='your_key_here'")
        print("🔄 Falling back to sample data generation...")
        return generate_sample_data_fallback(symbol)
    
    try:
        # Initialize Polygon client
        polygon_client = RESTClient(polygon_api_key)
        
        print(f"📊 Fetching data from {start_date} to {end_date}")
        
        # Get aggregates (daily bars) from Polygon
        aggs = []
        for agg in polygon_client.get_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_=start_date,
            to=end_date,
            limit=5000
        ):
            aggs.append(agg)
        
        if not aggs:
            raise ValueError(f"No data found for {symbol} from Polygon")
        
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
        
        print(f"✅ Downloaded {len(df)} days of real data from Polygon")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Failed to download data from Polygon: {e}")
        print("🔄 Falling back to sample data generation...")
        return generate_sample_data_fallback(symbol)

def generate_sample_data_fallback(symbol='SPY', days=300):
    """Generate sample OHLCV data as fallback"""
    print(f"📊 Generating {days} days of sample data for {symbol}...")
    
    # Set random seed for reproducibility
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
    best_params_file = f"best_parameters_{symbol}.json"
    
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
        useEmaFilter=False,  # Keep disabled for now
        emaPeriod=200,
        useSmaFilter=False,  # Keep disabled for now
        smaPeriod=200
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

def calculate_performance_metrics(results_df, symbol, initial_capital=10000):
    """Calculate comprehensive trading performance metrics"""
    
    # Get signal data
    long_entries = results_df['startLongTrade'].dropna()
    short_entries = results_df['startShortTrade'].dropna()
    long_exits = results_df['endLongTrade'].dropna()
    short_exits = results_df['endShortTrade'].dropna()
    
    # Calculate individual trade returns
    long_trades = []
    short_trades = []
    
    # Match long entries with exits
    for entry_date, entry_price in long_entries.items():
        # Find the next exit after this entry
        future_exits = long_exits[long_exits.index > entry_date]
        if not future_exits.empty:
            exit_date = future_exits.index[0]
            exit_price = future_exits.iloc[0]
            return_pct = ((exit_price - entry_price) / entry_price) * 100
            days_held = (exit_date - entry_date).days
            long_trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return_pct': return_pct,
                'days_held': days_held
            })
    
    # Match short entries with exits
    for entry_date, entry_price in short_entries.items():
        # Find the next exit after this entry
        future_exits = short_exits[short_exits.index > entry_date]
        if not future_exits.empty:
            exit_date = future_exits.index[0]
            exit_price = future_exits.iloc[0]
            return_pct = ((entry_price - exit_price) / entry_price) * 100  # Inverted for short trades
            days_held = (exit_date - entry_date).days
            short_trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return_pct': return_pct,
                'days_held': days_held
            })
    
    all_trades = long_trades + short_trades
    
    if not all_trades:
        return None
    
    # Calculate metrics
    returns = [trade['return_pct'] for trade in all_trades]
    winning_trades = [r for r in returns if r > 0]
    losing_trades = [r for r in returns if r < 0]
    
    total_return = sum(returns)
    win_rate = len(winning_trades) / len(returns) * 100 if returns else 0
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    avg_return = np.mean(returns)
    
    # Risk metrics
    std_return = np.std(returns) if len(returns) > 1 else 0
    sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0
    
    # Max drawdown calculation
    cumulative_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = cumulative_returns - running_max
    max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
    
    # Profit factor
    total_wins = sum(winning_trades) if winning_trades else 0
    total_losses = abs(sum(losing_trades)) if losing_trades else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    # Trading frequency
    start_date = results_df.index[0]
    end_date = results_df.index[-1]
    total_days = (end_date - start_date).days
    trades_per_month = len(all_trades) / (total_days / 30.44) if total_days > 0 else 0
    
    # Average holding period
    avg_holding_days = np.mean([trade['days_held'] for trade in all_trades])
    
    # Buy & Hold comparison
    initial_price = results_df['close'].iloc[0]
    final_price = results_df['close'].iloc[-1]
    buy_hold_return = ((final_price - initial_price) / initial_price) * 100
    
    # Portfolio value calculations
    final_portfolio_value = initial_capital * (1 + total_return / 100)
    total_dollar_return = final_portfolio_value - initial_capital
    buy_hold_final_value = initial_capital * (1 + buy_hold_return / 100)
    buy_hold_dollar_return = buy_hold_final_value - initial_capital
    
    # Average dollar return per trade
    avg_dollar_return_per_trade = total_dollar_return / len(all_trades) if all_trades else 0
    
    return {
        'symbol': symbol,
        'total_trades': len(all_trades),
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'total_return': total_return,
        'avg_return_per_trade': avg_return,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'std_return': std_return,
        'trades_per_month': trades_per_month,
        'avg_holding_days': avg_holding_days,
        'buy_hold_return': buy_hold_return,
        'start_date': start_date,
        'end_date': end_date,
        'total_days': total_days,
        'all_trades': all_trades,
        # Portfolio value metrics
        'initial_capital': initial_capital,
        'final_portfolio_value': final_portfolio_value,
        'total_dollar_return': total_dollar_return,
        'avg_dollar_return_per_trade': avg_dollar_return_per_trade,
        'buy_hold_final_value': buy_hold_final_value,
        'buy_hold_dollar_return': buy_hold_dollar_return
    }

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
    
    # Win/Loss ratio
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

def main():
    """Main demo function"""
    print("🎯 Starting Advanced TA Lorentzian Classification Demo")
    print(f"   Working directory: {os.getcwd()}")
    
    # Download real market data
    symbol = os.getenv('SYMBOL', 'TSLA')
    start_date = os.getenv('BACKTESTING_START', '2024-01-31')
    end_date = os.getenv('BACKTESTING_END', '2024-12-31')
    initial_capital = float(os.getenv('INITIAL_CAPITAL', '10000'))
    use_optimized_params = os.getenv('USE_OPTIMIZED_PARAMS', 'true').lower() == 'true'
    
    print(f"📊 Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   Initial capital: ${initial_capital:,.2f}")
    print(f"   Use optimized parameters: {use_optimized_params}")
    
    df = download_real_data(symbol=symbol, start_date=start_date, end_date=end_date)
    
    # Load optimized parameters if available and requested
    optimized_params = None
    if use_optimized_params:
        print(f"\n🔍 Checking for optimized parameters...")
        optimized_params = load_optimized_parameters(symbol)
    
    # Create features (optimized or default)
    features = create_features_from_params(optimized_params)
    
    print(f"\n📈 Features for classification:")
    for i, feature in enumerate(features, 1):
        print(f"   {i}. {feature.type}({feature.param1}, {feature.param2})")
    
    if optimized_params:
        print(f"   🎯 Using optimized parameters!")
    
    # Run classification
    print(f"\n🧠 Running Lorentzian Classification...")
    
    try:
        from lorentzian_classification import LorentzianClassification, Settings, FilterSettings, KernelFilter
        
        # Convert to the format expected by advanced_ta (lowercase columns)
        df_for_classification = df.copy()
        df_for_classification.columns = df_for_classification.columns.str.lower()
        
        # Create settings and filters (optimized or default)
        settings = create_settings_from_params(optimized_params, df_for_classification['close'])
        filter_settings = create_filter_settings_from_params(optimized_params)
        
        print(f"\n⚙️  Classification Settings:")
        print(f"   Neighbors: {settings.neighborsCount}")
        print(f"   Max bars back: {settings.maxBarsBack}")
        print(f"   Dynamic exits: {settings.useDynamicExits}")
        
        if optimized_params:
            print(f"   🎯 Using optimized filter settings")
        else:
            print(f"   Filters: All disabled for more signals")
        
        lc = LorentzianClassification(df_for_classification, features, settings, filter_settings)
        
        print(f"✅ Classification completed successfully!")
        
        # Create results directory and plot file path
        results_dir = "results_logs"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = os.path.join(results_dir, f"lorentzian_plot_{symbol}_{timestamp}.jpg")
        
        print(f"📊 Generating plot...")
        try:
            lc.plot(plot_file)
            print(f"✅ Demo completed successfully!")
            print(f"   Plot saved to: {plot_file}")
        except ImportError as plot_error:
            if "mplfinance" in str(plot_error):
                print(f"⚠️  Plot generation skipped - missing dependency")
                print(f"   Install with: pip install mplfinance")
                print(f"✅ Demo completed (without plot)")
            else:
                raise plot_error
        
        # Get results and calculate performance metrics
        results = lc.data
        long_signals = results['startLongTrade'].notna().sum()
        short_signals = results['startShortTrade'].notna().sum()
        print(f"   Long signals: {long_signals}")
        print(f"   Short signals: {short_signals}")
        
        # Calculate and display comprehensive performance metrics
        print(f"\n📊 Calculating performance metrics...")
        metrics = calculate_performance_metrics(results, symbol, initial_capital)
        display_performance_report(metrics)
        
    except Exception as e:
        print(f"❌ Classification failed: {str(e)}")
        if "mplfinance" in str(e):
            print(f"💡 Install missing dependency: pip install mplfinance")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
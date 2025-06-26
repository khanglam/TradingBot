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

def main():
    """Main demo function"""
    print("🎯 Starting Advanced TA Lorentzian Classification Demo")
    print(f"   Working directory: {os.getcwd()}")
    
    # Download real market data
    symbol = os.getenv('SYMBOL', 'SPY')
    start_date = os.getenv('BACKTESTING_START', '2023-01-01')
    end_date = os.getenv('BACKTESTING_END', '2024-12-31')
    
    print(f"📊 Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Date range: {start_date} to {end_date}")
    
    df = download_real_data(symbol=symbol, start_date=start_date, end_date=end_date)
    
    # Define features for classification
    features = [
        Feature("RSI", 14, 1),  # Reduced smoothing for more signals
        Feature("WT", 9, 10),   # Slightly more responsive
        Feature("CCI", 14, 1),  # Reduced smoothing for more signals
    ]
    
    print(f"\n📈 Features for classification:")
    for i, feature in enumerate(features, 1):
        print(f"   {i}. {feature.type}({feature.param1}, {feature.param2})")
    
    # Run classification
    print(f"\n🧠 Running Lorentzian Classification...")
    
    try:
        from lorentzian_classification import LorentzianClassification, Settings, FilterSettings, KernelFilter
        
        # Convert to the format expected by advanced_ta (lowercase columns)
        df_for_classification = df.copy()
        df_for_classification.columns = df_for_classification.columns.str.lower()
        
        # Define more balanced settings for better signal generation (AFTER data conversion)
        settings = Settings(
            source=df_for_classification['close'],  # Required: source data for classification
            neighborsCount=6,           # Reduced from default 8 for more signals
            maxBarsBack=1500,          # Keep reasonable history
            useDynamicExits=True,      # Enable dynamic exits
            useEmaFilter=False,        # Disable EMA filter
            emaPeriod=200,
            useSmaFilter=False,        # Disable SMA filter  
            smaPeriod=200
        )
        
        # Relaxed filter settings  
        kernel_filter = KernelFilter(
            useKernelSmoothing=False,
            lookbackWindow=8,
            relativeWeight=8.0,
            regressionLevel=25,
            crossoverLag=2
        )
        
        filter_settings = FilterSettings(
            useVolatilityFilter=False,  # Disable volatility filter
            useRegimeFilter=False,      # Disable regime filter
            useAdxFilter=False,         # Disable ADX filter
            regimeThreshold=0.0,
            adxThreshold=20,
            kernelFilter=kernel_filter
        )
        
        print(f"\n⚙️  Classification Settings:")
        print(f"   Neighbors: {settings.neighborsCount}")
        print(f"   Max bars back: {settings.maxBarsBack}")
        print(f"   Dynamic exits: {settings.useDynamicExits}")
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
        
        # Show some basic stats
        results = lc.data
        long_signals = results['startLongTrade'].notna().sum()
        short_signals = results['startShortTrade'].notna().sum()
        print(f"   Long signals: {long_signals}")
        print(f"   Short signals: {short_signals}")
        
    except Exception as e:
        print(f"❌ Classification failed: {str(e)}")
        if "mplfinance" in str(e):
            print(f"💡 Install missing dependency: pip install mplfinance")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
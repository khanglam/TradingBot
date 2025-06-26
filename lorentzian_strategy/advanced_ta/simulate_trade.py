"""
Trading Simulation Engine
========================

This module provides a reusable trading simulation engine that mimics the logic
from AdvancedLorentzianStrategy.py. It can be used by both optimize_parameters.py
and run_advanced_ta.py to ensure consistent and realistic backtesting.

The simulation follows the exact same logic as the Lumibot strategy:
- Single position tracking (can't have multiple open positions)
- Proper entry/exit signal handling
- Realistic portfolio value calculations
- Cash management and position sizing
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

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

@dataclass
class Trade:
    """Represents a completed trade"""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # 'long' or 'short'
    return_pct: float
    return_dollars: float
    days_held: int
    reason: str  # 'signal' or 'end_of_period'

@dataclass
class Position:
    """Represents an open position"""
    entry_date: datetime
    entry_price: float
    quantity: int
    side: str  # 'long' or 'short'
    
    def current_value(self, current_price: float) -> float:
        """Calculate current value of position"""
        if self.side == 'long':
            return self.quantity * current_price
        else:  # short
            return self.quantity * (2 * self.entry_price - current_price)
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        if self.side == 'long':
            return self.quantity * (current_price - self.entry_price)
        else:  # short
            return self.quantity * (self.entry_price - current_price)
    
    def unrealized_return_pct(self, current_price: float) -> float:
        """Calculate unrealized return percentage"""
        if self.side == 'long':
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:  # short
            return ((self.entry_price - current_price) / self.entry_price) * 100

class TradingSimulator:
    """
    Trading simulation engine that mimics AdvancedLorentzianStrategy logic
    """
    
    def __init__(self, initial_capital: float = 10000, position_size_pct: float = 0.95, 
                 cash_buffer: float = 1000, enable_short_selling: bool = False):
        """
        Initialize trading simulator
        
        Args:
            initial_capital: Starting capital
            position_size_pct: Percentage of available cash to use for positions (0.95 = 95%)
            cash_buffer: Minimum cash to keep in account
            enable_short_selling: Whether to allow short selling
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.cash_buffer = cash_buffer
        self.enable_short_selling = enable_short_selling
        
        # Reset simulation state
        self.reset()
    
    def reset(self):
        """Reset simulation to initial state"""
        self.cash = self.initial_capital
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.portfolio_values: List[float] = []
        self.dates: List[datetime] = []
        
    def simulate_trading(self, df: pd.DataFrame, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Simulate trading based on signals, following AdvancedLorentzianStrategy logic
        
        Args:
            df: OHLCV data with DatetimeIndex
            signals_df: DataFrame with trading signals (start_long, start_short columns)
            
        Returns:
            Dictionary with simulation results and performance metrics
        """
        self.reset()
        
        # Check for signal columns and convert if needed
        if 'start_long' in signals_df.columns and 'start_short' in signals_df.columns:
            # Already in correct format
            pass
        elif 'isNewBuySignal' in signals_df.columns and 'isNewSellSignal' in signals_df.columns:
            # Convert from LorentzianClassification boolean format
            signals_df = signals_df.copy()
            signals_df['start_long'] = signals_df['isNewBuySignal']
            signals_df['start_short'] = signals_df['isNewSellSignal']
        elif 'startLongTrade' in signals_df.columns and 'startShortTrade' in signals_df.columns:
            # Convert from LorentzianClassification price format
            signals_df = signals_df.copy()
            signals_df['start_long'] = signals_df['startLongTrade'].notna()
            signals_df['start_short'] = signals_df['startShortTrade'].notna()
        else:
            raise ValueError("signals_df must have 'start_long'/'start_short', 'isNewBuySignal'/'isNewSellSignal', or 'startLongTrade'/'startShortTrade' columns")
        
        # Align dataframes
        common_index = df.index.intersection(signals_df.index)
        df_aligned = df.loc[common_index]
        signals_aligned = signals_df.loc[common_index]
        
        if is_debug():
            print(f"🔄 Simulating trading on {len(df_aligned)} data points...")
        
        # Process each day
        for date, row in df_aligned.iterrows():
            current_price = row['close']
            
            # Get signals for this date
            if date in signals_aligned.index:
                start_long = signals_aligned.loc[date, 'start_long']
                start_short = signals_aligned.loc[date, 'start_short']
            else:
                start_long = False
                start_short = False
            
            # Process trading logic (mimics AdvancedLorentzianStrategy.on_trading_iteration)
            self._process_trading_signals(date, current_price, start_long, start_short)
            
            # Record portfolio value
            portfolio_value = self._calculate_portfolio_value(current_price)
            self.portfolio_values.append(portfolio_value)
            self.dates.append(date)
        
        # Close any open position at the end
        if self.position is not None:
            final_date = df_aligned.index[-1]
            final_price = df_aligned.iloc[-1]['close']
            self._close_position(final_date, final_price, 'end_of_period')
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(df_aligned)
        
        return {
            'trades': self.trades,
            'portfolio_values': self.portfolio_values,
            'dates': self.dates,
            'final_capital': self.cash + (self.position.current_value(final_price) if self.position else 0),
            'metrics': metrics
        }
    
    def _process_trading_signals(self, date: datetime, price: float, start_long: bool, start_short: bool):
        """Process trading signals for a single day (enhanced to use sell signals for exits)"""
        
        # Handle sell signals first (close long positions)
        if start_short and self.position is not None and self.position.side == 'long':
            self._close_position(date, price, 'sell_signal')
        
        # Handle buy signals (open new long positions)
        if start_long and self.position is None:
            self._open_long_position(date, price)
        
        # Short signal logic (only if short selling is enabled and no position exists)
        if start_short and self.enable_short_selling and self.position is None:
            self._open_short_position(date, price)
    
    def _open_long_position(self, date: datetime, price: float):
        """Open a long position"""
        # Calculate position size (mimics AdvancedLorentzianStrategy logic)
        available_cash = max(0, self.cash - self.cash_buffer)
        position_value = available_cash * self.position_size_pct
        quantity = int(position_value / price)
        
        if quantity > 0:
            cost = quantity * price
            if cost <= self.cash:
                self.cash -= cost
                self.position = Position(
                    entry_date=date,
                    entry_price=price,
                    quantity=quantity,
                    side='long'
                )
                if is_debug():
                    print(f"   📈 OPENED long: {quantity} shares @ ${price:.2f} on {date.date()}")
            else:
                if is_debug():
                    print(f"   ❌ Insufficient cash for long position: need ${cost:.2f}, have ${self.cash:.2f}")
    
    def _open_short_position(self, date: datetime, price: float):
        """Open a short position (if enabled)"""
        if not self.enable_short_selling:
            return
        
        # For short selling, we use portfolio value instead of just cash
        portfolio_value = self.cash
        position_value = portfolio_value * self.position_size_pct
        quantity = int(position_value / price)
        
        if quantity > 0:
            # In short selling, we receive cash upfront
            proceeds = quantity * price
            self.cash += proceeds
            self.position = Position(
                entry_date=date,
                entry_price=price,
                quantity=quantity,
                side='short'
            )
            if is_debug():
                print(f"   📉 OPENED short: {quantity} shares @ ${price:.2f} on {date.date()}")
    
    def _close_position(self, date: datetime, price: float, reason: str):
        """Close the current position"""
        if self.position is None:
            return
        
        if self.position.side == 'long':
            # Sell long position
            proceeds = self.position.quantity * price
            self.cash += proceeds
            
        else:  # short position
            # Buy to cover short position
            cost = self.position.quantity * price
            self.cash -= cost
        
        # Calculate trade metrics
        return_dollars = self.position.unrealized_pnl(price)
        return_pct = self.position.unrealized_return_pct(price)
        days_held = (date - self.position.entry_date).days
        
        # Record trade
        trade = Trade(
            entry_date=self.position.entry_date,
            exit_date=date,
            entry_price=self.position.entry_price,
            exit_price=price,
            quantity=self.position.quantity,
            side=self.position.side,
            return_pct=return_pct,
            return_dollars=return_dollars,
            days_held=days_held,
            reason=reason
        )
        self.trades.append(trade)
        
        if is_debug():
            print(f"   🔚 CLOSED {self.position.side}: {self.position.quantity} shares @ ${price:.2f} on {date.date()}")
            print(f"      Return: {return_pct:+.2f}% (${return_dollars:+,.2f}) | Days: {days_held}")
        
        # Clear position
        self.position = None
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value"""
        portfolio_value = self.cash
        if self.position is not None:
            portfolio_value += self.position.current_value(current_price)
        return portfolio_value
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_return': 0,
                'win_rate': 0,
                'avg_return': 0,
                'final_portfolio_value': self.cash,
                'buy_hold_return': 0
            }
        
        # Trade statistics
        returns = [trade.return_pct for trade in self.trades]
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r < 0]
        
        total_return = sum(returns)
        win_rate = len(winning_trades) / len(returns) * 100
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        avg_return = np.mean(returns)
        
        # Risk metrics
        std_return = np.std(returns) if len(returns) > 1 else 0
        sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0
        
        # Max drawdown
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        rolling_max = portfolio_series.expanding().max()
        drawdowns = (portfolio_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdowns.min()
        
        # Profit factor
        total_wins = sum(winning_trades) if winning_trades else 0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
        
        # Portfolio metrics
        final_portfolio_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_capital
        total_dollar_return = final_portfolio_value - self.initial_capital
        
        # Buy & hold comparison
        initial_price = df.iloc[0]['close']
        final_price = df.iloc[-1]['close']
        buy_hold_return = ((final_price - initial_price) / initial_price) * 100
        buy_hold_final_value = self.initial_capital * (1 + buy_hold_return / 100)
        
        # Trading frequency
        start_date = df.index[0]
        end_date = df.index[-1]
        total_days = (end_date - start_date).days
        trades_per_month = len(self.trades) / (total_days / 30.44) if total_days > 0 else 0
        
        # Average holding period
        avg_holding_days = np.mean([trade.days_held for trade in self.trades])
        
        return {
            'symbol': 'SIMULATION',
            'total_trades': len(self.trades),
            'long_trades': len([t for t in self.trades if t.side == 'long']),
            'short_trades': len([t for t in self.trades if t.side == 'short']),
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
            'all_trades': self.trades,
            # Portfolio value metrics
            'initial_capital': self.initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'total_dollar_return': total_dollar_return,
            'avg_dollar_return_per_trade': total_dollar_return / len(self.trades),
            'buy_hold_final_value': buy_hold_final_value,
            'buy_hold_dollar_return': buy_hold_final_value - self.initial_capital
        }

def run_trading_simulation(df: pd.DataFrame, features: List, settings, filter_settings, 
                          initial_capital: float = 10000, enable_short_selling: bool = False) -> Dict[str, Any]:
    """
    Convenience function to run a complete trading simulation
    
    Args:
        df: OHLCV data with lowercase columns
        features: List of Feature objects
        settings: ClassifierSettings object
        filter_settings: FilterSettings object
        initial_capital: Starting capital
        enable_short_selling: Whether to allow short selling
        
    Returns:
        Dictionary with simulation results and performance metrics
    """
    # Import here to avoid circular imports
    from classifier import LorentzianClassification
    
    # Run classification
    lc = LorentzianClassification(df, features, settings, filter_settings)
    
    # Create simulator
    simulator = TradingSimulator(
        initial_capital=initial_capital,
        enable_short_selling=enable_short_selling
    )
    
    # Run simulation
    results = simulator.simulate_trading(df, lc.data)
    
    return results 
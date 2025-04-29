import pandas as pd
import numpy as np
from typing import Callable, Dict, Any

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate
    return np.sqrt(252) * excess.mean() / (excess.std() + 1e-9)

def max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()

def run_backtest(df: pd.DataFrame, signal_func: Callable[[pd.DataFrame], pd.Series],
                 initial_balance: float = 10000, fee: float = 0.001) -> Dict[str, Any]:
    df = df.copy()
    df['signal'] = signal_func(df)
    balance = initial_balance
    position = 0
    position_type = 0  # 0: no position, 1: long, -1: short
    equity_curve = []
    trade_log = []
    prev_price = df['close'].iloc[0]
    for i in range(1, len(df)):
        signal = df['signal'].iloc[i]
        price = df['close'].iloc[i]
        # Enter long
        if signal == 1 and position_type == 0:
            position = balance / price
            position_type = 1
            balance = 0
            trade_log.append({'type': 'buy', 'price': price, 'qty': position, 'index': i})
        # Close short and go long
        elif signal == 1 and position_type == -1:
            balance += position * (prev_price - price)  # profit from short
            trade_log.append({'type': 'cover', 'price': price, 'qty': position, 'index': i})
            position = balance / price
            trade_log.append({'type': 'buy', 'price': price, 'qty': position, 'index': i})
            position_type = 1
            balance = 0
        # Enter short
        elif signal == -1 and position_type == 0:
            position = balance / price
            position_type = -1
            balance = 0
            trade_log.append({'type': 'short', 'price': price, 'qty': position, 'index': i})
        # Close long and go short
        elif signal == -1 and position_type == 1:
            balance = position * price * (1 - fee)
            trade_log.append({'type': 'sell', 'price': price, 'qty': position, 'index': i})
            position = balance / price
            trade_log.append({'type': 'short', 'price': price, 'qty': position, 'index': i})
            position_type = -1
            balance = 0
        # Close long
        elif signal == -1 and position_type == 1:
            balance = position * price * (1 - fee)
            trade_log.append({'type': 'sell', 'price': price, 'qty': position, 'index': i})
            position = 0
            position_type = 0
        # Close short
        elif signal == 1 and position_type == -1:
            balance += position * (prev_price - price)  # profit from short
            trade_log.append({'type': 'cover', 'price': price, 'qty': position, 'index': i})
            position = 0
            position_type = 0
        prev_price = price
        # Calculate equity
        if position_type == 1:
            equity = position * price
        elif position_type == -1:
            equity = balance + position * (prev_price - price)
        else:
            equity = balance
        equity_curve.append(equity)
    returns = pd.Series(equity_curve).pct_change().dropna()
    metrics = {
        'final_equity': equity_curve[-1] if equity_curve else initial_balance,
        'sharpe': sharpe_ratio(returns),
        'max_drawdown': max_drawdown(pd.Series(equity_curve)),
        'total_return': (equity_curve[-1] / initial_balance - 1) if equity_curve else 0,
        'trade_log': trade_log,
        'equity_curve': equity_curve
    }
    return metrics

import datetime
from typing import List, Dict

class PortfolioManager:
    def __init__(self, base_currency: str = 'USDT', initial_balance: float = 10000):
        self.base_currency = base_currency
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: List[Dict] = []  # [{'symbol': ..., 'qty': ..., 'entry_price': ..., 'side': ..., 'open_time': ...}]
        self.closed_positions: List[Dict] = []

    def open_position(self, symbol: str, qty: float, entry_price: float, side: str):
        pos = {
            'symbol': symbol,
            'qty': qty,
            'entry_price': entry_price,
            'side': side,
            'open_time': datetime.datetime.now(),
        }
        self.positions.append(pos)
        self.balance -= qty * entry_price

    def close_position(self, symbol: str, qty: float, exit_price: float):
        for pos in self.positions:
            if pos['symbol'] == symbol and pos['qty'] == qty:
                profit = (exit_price - pos['entry_price']) * qty
                self.balance += qty * exit_price
                pos['exit_price'] = exit_price
                pos['close_time'] = datetime.datetime.now()
                pos['profit'] = profit
                self.closed_positions.append(pos)
                self.positions.remove(pos)
                return profit
        return 0

    def get_portfolio_value(self, latest_prices: Dict[str, float]) -> float:
        value = self.balance
        for pos in self.positions:
            value += pos['qty'] * latest_prices.get(pos['symbol'], pos['entry_price'])
        return value

    def get_open_positions(self):
        return self.positions

    def get_closed_positions(self):
        return self.closed_positions

    def get_balance(self):
        return self.balance

    def reset(self):
        self.balance = self.initial_balance
        self.positions = []
        self.closed_positions = []

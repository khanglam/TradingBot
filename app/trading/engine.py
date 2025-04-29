import pandas as pd
from typing import Dict, List, Optional
import uuid
import datetime

class Position:
    def __init__(self, symbol: str, qty: float, entry_price: float, side: str):
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.qty = qty
        self.entry_price = entry_price
        self.side = side  # 'long' or 'short'
        self.open_time = datetime.datetime.now()
        self.close_time = None
        self.exit_price = None
        self.profit = 0.0

class PaperTradingEngine:
    def __init__(self, initial_balance: float = 10000):
        self.balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.trade_history: List[dict] = []

    def buy(self, symbol: str, price: float, qty: float):
        cost = price * qty
        if cost > self.balance:
            raise Exception("Insufficient balance for buy order.")
        self.balance -= cost
        pos = Position(symbol, qty, price, 'long')
        self.positions[pos.id] = pos
        self.trade_history.append({
            'type': 'buy', 'symbol': symbol, 'price': price, 'qty': qty, 'time': datetime.datetime.now()
        })
        return pos.id

    def sell(self, position_id: str, price: float):
        pos = self.positions.get(position_id)
        if not pos:
            raise Exception("Position not found.")
        profit = (price - pos.entry_price) * pos.qty
        self.balance += price * pos.qty
        pos.exit_price = price
        pos.close_time = datetime.datetime.now()
        pos.profit = profit
        self.closed_positions.append(pos)
        del self.positions[position_id]
        self.trade_history.append({
            'type': 'sell', 'symbol': pos.symbol, 'price': price, 'qty': pos.qty, 'profit': profit, 'time': datetime.datetime.now()
        })
        return profit

    def get_open_positions(self):
        return list(self.positions.values())

    def get_closed_positions(self):
        return self.closed_positions

    def get_balance(self):
        return self.balance

    def get_trade_history(self):
        return self.trade_history

    def reset(self, initial_balance: Optional[float] = None):
        self.positions = {}
        self.closed_positions = []
        self.trade_history = []
        if initial_balance is not None:
            self.balance = initial_balance

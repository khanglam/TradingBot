import pandas as pd
from typing import Dict, Any

class SampleMovingAverageStrategy:
    """
    Simple moving average crossover strategy for demonstration.
    """
    def __init__(self, fast: int = 10, slow: int = 30):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        df['fast_ma'] = df['close'].rolling(self.fast).mean()
        df['slow_ma'] = df['close'].rolling(self.slow).mean()
        df['signal'] = 0
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1
        return df['signal']

    def params(self) -> Dict[str, Any]:
        return {'fast': self.fast, 'slow': self.slow}

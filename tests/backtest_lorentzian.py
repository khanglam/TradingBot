import pandas as pd
from advanced_ta import LorentzianClassification
from backtesting import Backtest, Strategy
import warnings

# Suppress only the specific warning from backtesting.py about insufficient margin
warnings.filterwarnings(
    "ignore",
    message=".*Broker canceled the relative-sized order due to insufficient margin.*"
)

# Load and prepare your data
# Adjust path if needed

df = pd.read_csv('TSLA_daily_data.csv')
cols = ['open', 'high', 'low', 'close', 'volume', 'date']
df = df[[c for c in df.columns if c.lower() in cols]]
df.columns = [c.lower() for c in df.columns]
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

# Run Lorentzian Classification
lc = LorentzianClassification(df)
signal = lc.data['signal'].values  # Assumes 1=long, -1=short, 0=flat/hold

class LorentzianStrategy(Strategy):
    neighborsCount = 8  # Default value, will be optimized
    commission = 0.002  # Default value, will be optimized
    maxBarsBack = 1500  # Default value, will be optimized
    useDynamicExits = True  # Default value, will be optimized
    useVolatilityFilter = True  # Default value, will be optimized
    useRegimeFilter = True  # Default value, will be optimized
    useAdxFilter = True  # Default value, will be optimized
    regimeThreshold = 0  # Default value, will be optimized
    adxThreshold = 10  # Default value, will be optimized
    useKernelSmoothing = True  # Default value, will be optimized
    lookbackWindow = 10  # Default value, will be optimized
    relativeWeight = 4.0  # Default value, will be optimized
    regressionLevel = 10  # Default value, will be optimized
    crossoverLag = 1  # Default value, will be optimized
    useEmaFilter = True  # Default value, will be optimized
    emaPeriod = 50  # Default value, will be optimized
    useSmaFilter = True  # Default value, will be optimized
    smaPeriod = 50  # Default value, will be optimized
    num_Bars_var = 2  # Default value, will be optimized

    def init(self):
        # Convert columns to lowercase for LorentzianClassification
        lc_df = self.data.df.copy()
        lc_df.columns = [c.lower() for c in lc_df.columns]
        # Ensure all required settings are passed explicitly
        lc = LorentzianClassification(
            lc_df,
            settings=LorentzianClassification.Settings(
                neighborsCount=self.neighborsCount,
                source='close'  # Set to the default or desired source column
            )
        )
        self.signal = lc.data['signal'].values

    def next(self):
        idx = len(self.data) - 1  # Current bar index
        min_equity = 100  # Minimum equity threshold to keep trading
        if self.equity < min_equity:
            if self.position:
                self.position.close()
            return  # Stop trading if equity is too low
        # Use 10% of available equity for each trade
        trade_size = 0.1
        if self.signal[idx] == 1 and not self.position.is_long:
            self.buy(size=trade_size)
        elif self.signal[idx] == -1 and not self.position.is_short:
            self.sell(size=trade_size)
        elif self.signal[idx] == 0:
            self.position.close()

# Prepare data for backtesting
bt_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
bt_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# Run optimization for neighborsCount
bt = Backtest(bt_df, LorentzianStrategy, cash=10000, commission=.002)

num_iterations = 10  

stats, best_params = bt.optimize(
    neighborsCount=range(2, 20, 2),
    commission=[0.001, 0.002, 0.003],
    maxBarsBack=range(1500, 3001, 500),
    useDynamicExits=[True, False],
    useVolatilityFilter=[True, False],
    useRegimeFilter=[True, False],
    useAdxFilter=[True, False],
    regimeThreshold=[-0.2, -0.1, 0, 0.1],
    adxThreshold=[10, 20, 30],
    useKernelSmoothing=[True, False],
    lookbackWindow=range(5, 15),
    relativeWeight=[4.0, 8.0, 12.0],
    regressionLevel=[10, 25, 50],
    crossoverLag=[1, 2, 3],
    useEmaFilter=[True, False],
    emaPeriod=[50, 100, 200],
    useSmaFilter=[True, False],
    smaPeriod=[50, 100, 200],
    num_Bars_var=range(2, 6),
    maximize='Return [%]',
    constraint=lambda param: param.neighborsCount < 15 or param.commission < 0.003,
    return_optimization=True,
    method='sambo',
    max_tries=num_iterations
)

# Print only the best parameters found
if hasattr(best_params, 'x') and hasattr(best_params, 'space'):
    # Try to extract parameter names and values from optimization result
    param_names = getattr(best_params.space, 'param_names', None)
    if param_names is not None:
        best_param_dict = dict(zip(param_names, best_params.x))
        print(best_param_dict)
    else:
        # Fallback: use the order of arguments in the optimize call
        fallback_param_names = [
            'neighborsCount', 'commission', 'maxBarsBack', 'useDynamicExits',
            'useVolatilityFilter', 'useRegimeFilter', 'useAdxFilter', 'regimeThreshold',
            'adxThreshold', 'useKernelSmoothing', 'lookbackWindow', 'relativeWeight',
            'regressionLevel', 'crossoverLag', 'useEmaFilter', 'emaPeriod', 'useSmaFilter',
            'smaPeriod', 'num_Bars_var'
        ]
        best_param_dict = dict(zip(fallback_param_names, best_params.x))
        print("Best parameters found:")
        max_key_len = max(len(str(k)) for k in best_param_dict.keys())
        for k in sorted(best_param_dict.keys()):
            print(f"  {k.ljust(max_key_len)} : {best_param_dict[k]}")
else:
    print(best_params)

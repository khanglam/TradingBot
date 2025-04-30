import pandas as pd
import optuna
from advanced_ta import LorentzianClassification
import matplotlib.pyplot as plt

#setup of the local folder
directory_path = 'C:/Users/rosti/Documents/Python ML/Re_start'

# Define an offset to modify the subset of data in each iteration
offset = 0

class LCSettings:
    def __init__(self):
        # Settings for LorentzianClassification
        self.source = 'close'
        self.neighborsCount = 8
        self.maxBarsBack = 2500
        self.useDynamicExits = False
        
        # FilterSettings
        self.useVolatilityFilter = False
        self.useRegimeFilter = False
        self.useAdxFilter = False
        self.regimeThreshold = -0.1
        self.adxThreshold = 20
        
        # KernelFilter
        self.useKernelSmoothing = False
        self.lookbackWindow = 8
        self.relativeWeight = 8.0
        self.regressionLevel = 25
        self.crossoverLag = 2
        
        # Features
        self.features = [
            ("RSI", 22, 7),
            #("RSI", 2, 3),
            #("CCI", 31, 2),
            #("ADX", 5, 2),
            #("RSI", 9, 2),
            #"MFI"
        ]
        
        # EMA Settings
        self.useEmaFilter = False
        self.emaPeriod = 200

        # SMA Settings
        self.useSmaFilter = False
        self.smaPeriod = 200

        #number of bars to hold
        self.num_Bars_var = 4

# Instantiate settings
settings = LCSettings()

# Load the stock data into a DataFrame
df = pd.read_csv(f'{directory_path}\\EURUSD_M15_50000_bars.csv')
print("Number of rows in df:", df.shape[0])

# Convert settings features to LorentzianClassification features
lc_features = []
for feature in settings.features:
    if isinstance(feature, tuple):
        lc_features.append(LorentzianClassification.Feature(*feature))
    else:
        lc_features.append(df[feature])

import pandas as pd

def backtest(df, settings):
    # Constants
    COMMISSION_PER_LOT = 2.5  # USD
    LOT_SIZE = 100000  # Standard lot size in forex
    TRADE_SIZE = 10000  # Size of the trade
    POINT_VALUE = TRADE_SIZE / LOT_SIZE  # Point value relative to the trade size
    COMMISSION = COMMISSION_PER_LOT * POINT_VALUE
    
    bars_to_hold = settings.num_Bars_var #bars to hold open trade is set up for the num_Bars_var which is LC variable for holding bars
    num_bars_back = settings.maxBarsBack 
    
    total_trades = 0
    successful_trades = 0
    total_profit_loss = 0
    trade_info_list = []  # List to collect trade info dictionaries

    # Check for necessary columns
    for column in ['isNewBuySignal', 'isNewSellSignal', 'time', 'close', 'low', 'high']:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

    # Subset the dataframe to only consider 'num_bars_back' rows
    subset_df = df.iloc[-num_bars_back:]

    #backtest loop
    for i, row in subset_df.iterrows():
        if i + bars_to_hold >= len(df):  # Skip the last "bars_to_hold" rows to avoid index out of range
            break

        trade_info = {}

        # Long Trade
        if row['isNewBuySignal']:
            trade_info = {
                'Entry Date': row['time'],
                'Exit Date': df.iloc[i + bars_to_hold]['time'],
                'Trade Type': 'Long',
                'Entry Price': row['close'],
                'Exit Price': df.iloc[i + bars_to_hold]['close'],
                'Drawdown': (row['close'] - df.iloc[i:i + bars_to_hold]['low'].min()) * TRADE_SIZE,
                'Profit/Loss': (df.iloc[i + bars_to_hold]['close'] - row['close']) * TRADE_SIZE - COMMISSION,
                'Successful': df.iloc[i + bars_to_hold]['close'] > row['close']
            }

        # Short Trade
        elif row['isNewSellSignal']:
            trade_info = {
                'Entry Date': row['time'],
                'Exit Date': df.iloc[i + bars_to_hold]['time'],
                'Trade Type': 'Short',
                'Entry Price': row['close'],
                'Exit Price': df.iloc[i + bars_to_hold]['close'],
                'Drawdown': (df.iloc[i:i + bars_to_hold]['high'].max() - row['close']) * TRADE_SIZE,
                'Profit/Loss': (row['close'] - df.iloc[i + bars_to_hold]['close']) * TRADE_SIZE - COMMISSION,
                'Successful': row['close'] > df.iloc[i + bars_to_hold]['close']
            }

        if trade_info:  # If trade_info dictionary is not empty
            # Update counters
            total_trades += 1
            if trade_info['Successful']:
                successful_trades += 1
            total_profit_loss += trade_info['Profit/Loss']
            trade_info_list.append(trade_info)  # Add the trade_info to the list

    # Create a DataFrame from the list of trade information
    trade_data = pd.DataFrame(trade_info_list)

    # Calculate success rate and handle division by zero
    success_rate = (successful_trades / total_trades * 100) if total_trades else 0

    return trade_data, success_rate

# Objective function for Optuna
def objective(trial):
    settings = LCSettings()

    # Using optuna to suggest values
    # UNCOMMENT THE LINE, WHICH YOU WANT TO USE

    settings.neighborsCount = trial.suggest_int("neighborsCount", 9, 20)
    #settings.maxBarsBack = trial.suggest_int("maxBarsBack", 1500, 5000, step=500)
    #settings.useDynamicExits = trial.suggest_categorical("useDynamicExits", [True, False])

    # FilterSettings
    settings.useVolatilityFilter = trial.suggest_categorical("useVolatilityFilter", [True, False])
    settings.useRegimeFilter = trial.suggest_categorical("useRegimeFilter", [True, False])
    settings.useAdxFilter = trial.suggest_categorical("useAdxFilter", [True, False])
    settings.regimeThreshold = trial.suggest_float("regimeThreshold", -2.0, 2.0, step=0.05)
    settings.adxThreshold = trial.suggest_int("adxThreshold", 10, 50)

    # KernelFilter
    settings.useKernelSmoothing = trial.suggest_categorical("useKernelSmoothing", [True, False])
    #settings.lookbackWindow = trial.suggest_int("lookbackWindow", 6, 20)
    #settings.relativeWeight = trial.suggest_float("relativeWeight", 1.0, 10.0, step=0.1)
    #settings.regressionLevel = trial.suggest_int("regressionLevel", 10, 100, step=5)
    #settings.crossoverLag = trial.suggest_int("crossoverLag", 1, 10)

    # EMA Settings
    settings.useEmaFilter = trial.suggest_categorical("useEmaFilter", [True, False])
    settings.emaPeriod = trial.suggest_int("emaPeriod", 50, 700, step=10)

    # SMA Settings
    settings.useSmaFilter = trial.suggest_categorical("useSmaFilter", [True, False])
    settings.smaPeriod = trial.suggest_int("smaPeriod", 50, 700, step=10)

    # Number of bars to hold
    settings.num_Bars_var = trial.suggest_int("num_Bars_var", 2, 9)

    
    # Suggest the number of features
    MAX_FEATURES = 6
    num_features = trial.suggest_int("num_features", 1, MAX_FEATURES)
    settings.features = []
    for i in range(num_features):
        #feature_type = trial.suggest_categorical(f"feature_type_{i}", ["ADX"])
        feature_type = trial.suggest_categorical(f"feature_type_{i}", ["RSI", "CCI", "ADX", "MFI", "WT"])
        
        if feature_type == "RSI":
            feature_param1 = trial.suggest_int(f"feature_param1_{i}", 80, 130)
            feature_param2 = trial.suggest_int(f"feature_param2_{i}", 60, 110)
        elif feature_type == "CCI":
            feature_param1 = trial.suggest_int(f"feature_param1_{i}", 2, 20)
            feature_param2 = trial.suggest_int(f"feature_param2_{i}", 2, 20)
        elif feature_type == "ADX":
            feature_param1 = trial.suggest_int(f"feature_param1_{i}", 80, 120)
            feature_param2 = trial.suggest_int(f"feature_param2_{i}", 2, 2)
        elif feature_type == "MFI":
            feature_param1 = trial.suggest_int(f"feature_param1_{i}", 2, 110)
            feature_param2 = trial.suggest_int(f"feature_param2_{i}", 2, 30)
        elif feature_type == "WT":
            feature_param1 = trial.suggest_int(f"feature_param1_{i}", 2, 110)
            feature_param2 = trial.suggest_int(f"feature_param2_{i}", 2, 30)

        settings.features.append((feature_type, feature_param1, feature_param2))

    # Convert settings features to LorentzianClassification features
    lc_features = [LorentzianClassification.Feature(*feature) for feature in settings.features]
    
    # Instantiate the LorentzianClassification with the features
    total_profit_loss_overall = 0
    total_trades_overall = 0
    successful_trades_overall = 0
    profitable = 0
    very_sucessfull = 1

    print("source:", settings.source)
    print("neighborsCount:", settings.neighborsCount)
    print("maxBarsBack:", settings.maxBarsBack)
    print("useDynamicExits:", settings.useDynamicExits)
    print("useEmaFilter:", settings.useEmaFilter)
    print("emaPeriod:", settings.emaPeriod)
    print("useSmaFilter:", settings.useSmaFilter)
    print("smaPeriod:", settings.smaPeriod)
    print("num_Bars_var:", settings.num_Bars_var)
    print("useVolatilityFilter:", settings.useVolatilityFilter)
    print("useRegimeFilter:", settings.useRegimeFilter)
    print("useAdxFilter:", settings.useAdxFilter)
    print("regimeThreshold:", settings.regimeThreshold)
    print("adxThreshold:", settings.adxThreshold)
    print("useKernelSmoothing:", settings.useKernelSmoothing)
    print("lookbackWindow:", settings.lookbackWindow)
    print("relativeWeight:", settings.relativeWeight)
    print("regressionLevel:", settings.regressionLevel)
    print("crossoverLag:", settings.crossoverLag)

    #Features printing
    for idx, feature in enumerate(lc_features):
        print(f"lc_features[{idx}]: Type={feature.type}, Param1={feature.param1}, Param2={feature.param2}")

    #Loop to iterate through different chunks can be adjusted for example: [0, 2000, 4000, 6000, 8000]
    for offset in [0, 2000, 4000]:
        # Adjust dataframe for current iteration and reset index
        iter_df = df.iloc[offset:offset + settings.maxBarsBack].copy().reset_index(drop=True)  
     
        # Instantiate the LorentzianClassification with the features
        lc = LorentzianClassification(
            iter_df,
            features=lc_features,
            settings=LorentzianClassification.Settings(
                source=settings.source,
                neighborsCount=settings.neighborsCount,
                maxBarsBack=settings.maxBarsBack,
                useDynamicExits=settings.useDynamicExits,
                useEmaFilter=settings.useEmaFilter, 
                emaPeriod=settings.emaPeriod,  
                useSmaFilter=settings.useSmaFilter,  
                smaPeriod=settings.smaPeriod,
                num_Bars_var=settings.num_Bars_var
            ),
            filterSettings=LorentzianClassification.FilterSettings(
                useVolatilityFilter=settings.useVolatilityFilter,
                useRegimeFilter=settings.useRegimeFilter,
                useAdxFilter=settings.useAdxFilter,
                regimeThreshold=settings.regimeThreshold,
                adxThreshold=settings.adxThreshold,
                kernelFilter=LorentzianClassification.KernelFilter(
                    useKernelSmoothing=settings.useKernelSmoothing,
                    lookbackWindow=settings.lookbackWindow,
                    relativeWeight=settings.relativeWeight,
                    regressionLevel=settings.regressionLevel,
                    crossoverLag=settings.crossoverLag
                )
            )
        )
        # Run your backtest on the lc.data
        trade_results, success_rate = backtest(df=lc.data, settings=settings)
        if not trade_results.empty:
            profit_loss = trade_results["Profit/Loss"].sum()
        else:
            profit_loss = 0  # or any other default value you deem appropriate

        print(f"Success Rate: {success_rate}%, Profit/Loss: {profit_loss}")
        if profit_loss>0:
            profitable +=1
        if success_rate>=54:
            very_sucessfull +=0.33

        if not trade_results.empty:
            total_profit_loss_overall += profit_loss
            total_trades_overall += len(trade_results)
            successful_trades_overall += trade_results[trade_results['Successful'] == True].shape[0]

    success_rate_overall = (successful_trades_overall / total_trades_overall) * 100
    
        # Create a dictionary with trial data
    trial_data = {
        **trial.params,
        'success_rate': success_rate_overall,
        'profit_loss': total_profit_loss_overall,
        'num_trades': total_trades_overall,
        'profitable_runs': profitable
    }       

    # Convert the dictionary to a DataFrame and append to all_trials_df
    global all_trials_df
    trial_data_df = pd.DataFrame([trial_data])
    all_trials_df = pd.concat([all_trials_df, trial_data_df], ignore_index=True)
  
    print(f"SUM: Success Rate: {success_rate_overall:.2f}%, Profit/Loss: {total_profit_loss_overall:.2f}, profitable: {profitable:.2f} and very succesfull: {very_sucessfull:.2f} ")
    
    if profitable == 3:
        goal = total_profit_loss_overall * very_sucessfull * profitable/2
    else:
        goal = total_profit_loss_overall
    print(f"SUM GOAL {goal:.2f}")

    return goal

all_trials_df = pd.DataFrame()
    
# Initialize the study

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5000)

all_trials_df.to_csv(f'{directory_path}/all_trials_data_5k.csv', index=False)

# Print out the optimal hyperparameters found.
print(study.best_params)

# Visualization
#optuna.visualization.plot_optimization_history(study).show()
#optuna.visualization.plot_param_importances(study).show()
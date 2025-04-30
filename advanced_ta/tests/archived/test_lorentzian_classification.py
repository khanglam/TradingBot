import pytest
import os
import shutil
import pandas as pd
from advanced_ta import LorentzianClassification
import yfinance as yf

# os.chdir('tests')  # Removed to avoid FileNotFoundError when already in 'tests' directory.


@pytest.fixture
def setup():
    try: os.mkdir('output')
    except FileExistsError: pass
    

def teardown():
    shutil.rmtree(os.path.join(os.getcwd(), 'output'), ignore_errors=True)


def load_data(file: str):
    df = pd.read_csv(file)
    df['date'] = pd.DatetimeIndex(df['date'])
    df.set_index("date", inplace=True)
    return df


def test_case1(setup):
    #df = load_data("data/NSE_NIFTY_MONTHLY.csv")
    data = yf.download("TSLA", start="2020-01-01", end="2024-01-01")
    
    # Extract the necessary columns and rename them to match the expected names
    df = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']

    # Add 'date' as a column based on the index of the DataFrame
    #df['date'] = df.index

    lc = LorentzianClassification(df)
    lc.dump('output/result1.csv')
    lc.plot('output/result1.jpg')

    df1 = load_data('output/result1.csv')
    df2 = load_data('expected/lc_case1.csv')

    #assert df1.equals(df2)


def test_case2(setup):
    df = load_data("data/NSE_NIFTY_MONTHLY.csv")

    lc = LorentzianClassification(
        df,
        features=[
            LorentzianClassification.Feature("RSI", 14, 2),  # f1
            LorentzianClassification.Feature("WT", 10, 11),  # f2
            LorentzianClassification.Feature("CCI", 20, 2),  # f3
            LorentzianClassification.Feature("ADX", 20, 2),  # f4
            LorentzianClassification.Feature("RSI", 9, 2),   # f5
        ],
        settings=LorentzianClassification.Settings(
            source=df['close'],
            neighborsCount=8,
            maxBarsBack=2000,
            useDynamicExits=False
        ),
        filterSettings=LorentzianClassification.FilterSettings(
            useVolatilityFilter=True,
            useRegimeFilter=True,
            useAdxFilter=False,
            regimeThreshold=-0.1,
            adxThreshold=20,
            kernelFilter=LorentzianClassification.KernelFilter()
        ))
    lc.dump('output/result2.csv')
    lc.plot('output/result2.jpg')

    df1 = load_data('output/result2.csv')
    df2 = load_data('expected/lc_case1.csv')

    assert df1.equals(df2)


def test_case3(setup):
    from advanced_ta import LorentzianClassification
    df = load_data("data/NSE_NIFTY_DAILY.csv")

    lc = LorentzianClassification(
        df,
        features=[
            df['f1'],  # f1
            df['f2'],  # f2
            df['f3'],  # f3
            df['f4'],  # f4
            df['f5'],  # f5
        ])
    lc.dump('output/result3.csv')
    lc.plot('output/result3.jpg')

    df1 = load_data('output/result3.csv')
    df2 = load_data('expected/lc_case3.csv')

    assert df1.equals(df2)

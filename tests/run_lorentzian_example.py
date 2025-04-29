import pandas as pd
from advanced_ta import LorentzianClassification

# Load sample data (adjust path if needed)
df = pd.read_csv('TSLA_daily_data.csv')

# Ensure columns are lowercase
cols = ['open', 'high', 'low', 'close', 'volume', 'date']
df = df[[c for c in df.columns if c.lower() in cols]]
df.columns = [c.lower() for c in df.columns]

# Set date as index if present and convert to DatetimeIndex
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

# Run Lorentzian Classification
lc = LorentzianClassification(df)
lc.dump('output/result.csv')
lc.plot('output/result.jpg')

print('Classification results saved to output/result.csv and output/result.jpg')

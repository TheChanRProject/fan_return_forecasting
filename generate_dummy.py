import pandas as pd
import numpy as np

# Generate 150 days of data
dates = pd.date_range(start="2023-01-01", periods=150)
prices = np.linspace(150, 200, 150) + np.random.normal(0, 2, 150)
df = pd.DataFrame({
    "Date": dates,
    "Open": prices,
    "High": prices + 1,
    "Low": prices - 1,
    "Close": prices,
    "Adj Close": prices,
    "Volume": 10000
})
df.to_csv("data/raw/AAPL.csv", index=False)

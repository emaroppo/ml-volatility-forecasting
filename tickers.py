import yfinance as yf
import json
import numpy as np
import pandas as pd

with open("data/tickers.json", "r") as f:
    tickers_symbols = json.load(f)

tickers_symbols_samples = tickers_symbols["communication-services"][
    "advertising-agencies"
]

tickers = yf.Tickers((" ").join(tickers_symbols_samples))

history = tickers.download("50y", interval="1d")
print(history.head())
history.columns = history.columns.swaplevel(0, 1)
history = history.sort_index(axis=1)
print(history.shape)
# create log returns column for each ticker
history = history.ffill()
for ticker in tickers_symbols_samples:
    history[ticker, "log_returns"] = np.log(
        history[ticker, "Close"] / history[ticker, "Close"].shift(1)
    )
print(history.head())
log_returns = history.xs("log_returns", axis=1, level=1)
log_returns = log_returns.dropna(axis=0, how="all")

volatility = pd.DataFrame()


weekly_volatility = log_returns.resample("W").std()

print(weekly_volatility)

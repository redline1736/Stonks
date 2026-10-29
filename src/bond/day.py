import yfinance as yf
import numpy as np

ticker_symbol = "^IRX"

df = yf.download(
    ticker_symbol,
    start="2025-01-01",
    end="2026-01-01",
    auto_adjust=True
)

# Extract closing prices (handles MultiIndex columns)
if isinstance(df.columns, __import__("pandas").MultiIndex):
    close_prices = df.xs("Close", axis=1, level=0)[ticker_symbol]
else:
    close_prices = df["Close"]

close_prices = close_prices.dropna()

# Create 2D array: [date, value]
data_array = np.column_stack((
    close_prices.index.strftime("%Y-%m-%d").to_numpy(),
    close_prices.to_numpy()
))

print(data_array.shape)
print(data_array)
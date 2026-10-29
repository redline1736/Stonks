import yfinance as yf

# 1. One stock: recent historical data
spy = yf.Ticker("SPY")
data = spy.history(period="5d")
print(data.head())

# 2. One stock: specific date range
data = spy.history(
    start="2022-01-03",
    end="2022-01-30",
    interval="1d"
)
print(data.head())

# 3. Multiple stocks: most recent month
data = yf.download(
    ["GOOG", "META"],
    period="1mo"
)
print(data.head())

# 4. Multiple stocks: specific date range
data = yf.download(
    ["GOOG", "META"],
    start="2022-01-03",
    end="2023-01-03",
    interval="1d",
    group_by="ticker"
)
print(data.head())

# get induvitial 

print(data["GOOG"].head())
print(data["META"].head())
# get closing 

print(data["GOOG"]["Close"])



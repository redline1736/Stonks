import yfinance as yf
# Create a Ticker object for the 30-year Treasury Yield
TYX = yf.Ticker("^TYX")

# Get current info dictionary
info = TYX.info
print("Current 30-year Treasury Yield:", info.get("regularMarketPrice"))

# Or pull historical data for recent yields
hist = TYX.history(period="5d")
print(hist)
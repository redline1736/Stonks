import yfinance as yf
"""• ^IRX: 3-month Treasury Bill
• ^FVX: 5-year Treasury Yield
• ^TNX: 10-year Treasury Yield
• ^TYX: 30-year Treasury Yield
"""
# Create a Ticker object for the 3-month Treasury Bill
IRX = yf.Ticker("^IRX")

# Get current info dictionary
info = IRX.info
print("Current 3-month Treasury Bill:", info.get("regularMarketPrice"))

# Or pull historical data for recent yields
hist = IRX.history(period="5d")
print(hist[["Close"]])

# Create a Ticker object for the 5-year Treasury Yield
FVX = yf.Ticker("^FVX")

# Get current info dictionary
info = FVX.info
print("Current 5-year Treasury Yield:", info.get("regularMarketPrice"))

# Or pull historical data for recent yields
hist = FVX.history(period="5d")
print(hist[["Close"]])

# Create a Ticker object for the 10-year Treasury yield
tnx = yf.Ticker("^TNX")

# Get current info dictionary
info = tnx.info
print("Current 10-Year Treasury Yield:", info.get("regularMarketPrice"))

# Or pull historical data for recent yields
hist = tnx.history(period="5d")
print(hist[["Close"]])

# Create a Ticker object for the 30-year Treasury Yield
TYX = yf.Ticker("^TYX")

# Get current info dictionary
info = TYX.info
print("Current 30-year Treasury Yield:", info.get("regularMarketPrice"))

# Or pull historical data for recent yields
hist = TYX.history(period="5d")
print(hist[["Close"]])

# Install dependencies:
# python -m pip install yfinance pandas pandas_datareader matplotlib

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web

# -----------------------------
# Settings
# -----------------------------
start_date = "2015-01-01"
end_date = (
    pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
).strftime("%Y-%m-%d")

# -----------------------------
# Download PCE Price Index
# -----------------------------
pce = web.DataReader(
    "PCEPI",
    "fred",
    start_date,
    end_date,
)["PCEPI"]

pce.index = pd.to_datetime(pce.index)
pce = pce.resample("MS").last()

# Calculate actual inflation rates
# Year-over-year percentage change
pce_inflation = pce.pct_change(periods=12) * 100

# Optional: core PCE, excluding food and energy
core_pce = web.DataReader(
    "PCEPILFE",
    "fred",
    start_date,
    end_date,
)["PCEPILFE"]

core_pce.index = pd.to_datetime(core_pce.index)
core_pce = core_pce.resample("MS").last()
core_inflation = core_pce.pct_change(periods=12) * 100

# -----------------------------
# Download Treasury yields
# FRED series are already in percentage points
# -----------------------------
fred_yields = web.DataReader(
    ["DGS5", "DGS10", "DGS30"],
    "fred",
    start_date,
    end_date,
)

fred_yields = fred_yields.resample("MS").mean()

fred_yields = fred_yields.rename(columns={
    "DGS5": "5-Year Treasury Yield",
    "DGS10": "10-Year Treasury Yield",
    "DGS30": "30-Year Treasury Yield",
})

# -----------------------------
# Download MOVE Index
# -----------------------------
move_raw = yf.download(
    "^MOVE",
    start=start_date,
    end=end_date,
    auto_adjust=False,
    progress=False,
)

if move_raw.empty:
    raise RuntimeError("Yahoo Finance returned no MOVE Index data.")

if isinstance(move_raw.columns, pd.MultiIndex):
    if "Close" in move_raw.columns.get_level_values(0):
        move = move_raw.xs("Close", axis=1, level=0).iloc[:, 0]
    else:
        raise RuntimeError("Could not find MOVE closing prices.")
else:
    move = move_raw["Close"]

move.index = pd.to_datetime(move.index)
move = move.resample("MS").mean()
move.name = "MOVE Index"

# -----------------------------
# Combine monthly data
# -----------------------------
data = pd.concat(
    [
        pce.rename("PCE Price Index"),
        pce_inflation.rename("PCE Inflation"),
        core_inflation.rename("Core PCE Inflation"),
        fred_yields,
        move,
    ],
    axis=1,
).sort_index()

# Remove the initial 12 months needed for YoY inflation
data = data.loc[pd.Timestamp(start_date) + pd.DateOffset(months=12):]

if data.empty:
    raise RuntimeError("No overlapping data found.")

# -----------------------------
# Plot: PCE inflation vs. 10-year Treasury yield
# -----------------------------
plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(
    data.index,
    data["PCE Inflation"],
    label="PCE Inflation (YoY)",
    color="#2563eb",
    linewidth=2.5,
)

ax.plot(
    data.index,
    data["10-Year Treasury Yield"],
    label="10-Year Treasury Yield",
    color="#ea580c",
    linewidth=2.5,
)

ax.set_title(
    "PCE Inflation vs. 10-Year US Treasury Yield",
    fontsize=16,
    fontweight="bold",
    pad=15,
)

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Annual Rate (%)", fontsize=12)

ax.legend(loc="best", fontsize=11)
ax.grid(True, alpha=0.3)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# Save the simplified data
data[["PCE Inflation", "10-Year Treasury Yield"]].to_csv(
    "pce_inflation_vs_10yr_treasury.csv"
)

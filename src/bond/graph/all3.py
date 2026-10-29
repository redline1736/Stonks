
# Install dependencies:
# python -m pip install yfinance pandas pandas_datareader matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as web

# -----------------------------
# Settings
# -----------------------------
start_date = "2015-01-01"
end_date = (
    pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
).strftime("%Y-%m-%d")

# -----------------------------
# 1. PCE inflation from FRED
# -----------------------------
pce = web.DataReader(
    "PCEPI",
    "fred",
    start_date,
    end_date,
)["PCEPI"]

pce.index = pd.to_datetime(pce.index)
pce = pce.resample("MS").last()

# Year-over-year inflation (%)
pce_inflation = pce.pct_change(12) * 100
pce_inflation.name = "PCE Inflation (YoY)"

# -----------------------------
# 2. 10-year Treasury yield from FRED
# -----------------------------
yield_data = web.DataReader(
    "DGS10",
    "fred",
    start_date,
    end_date,
)["DGS10"]

yield_data.index = pd.to_datetime(yield_data.index)

# Monthly average of daily yields (%)
treasury_yield = yield_data.resample("MS").mean()
treasury_yield.name = "10-Year Treasury Yield"

# -----------------------------
# 3. ZN Treasury futures from Yahoo Finance
# -----------------------------
zn_raw = yf.download(
    "ZN=F",
    start=start_date,
    end=end_date,
    auto_adjust=False,
    progress=False,
)

if zn_raw.empty:
    raise RuntimeError(
        "Yahoo Finance returned no ZN futures data. "
        "Check the ticker and your internet connection."
    )

# Handle different yfinance column formats
if isinstance(zn_raw.columns, pd.MultiIndex):
    if "Close" in zn_raw.columns.get_level_values(0):
        zn_close = zn_raw.xs(
            "Close", axis=1, level=0
        ).iloc[:, 0]
    elif "Adj Close" in zn_raw.columns.get_level_values(0):
        zn_close = zn_raw.xs(
            "Adj Close", axis=1, level=0
        ).iloc[:, 0]
    else:
        raise RuntimeError(
            "Could not find ZN futures closing prices."
        )
else:
    if "Close" in zn_raw.columns:
        zn_close = zn_raw["Close"]
    elif "Adj Close" in zn_raw.columns:
        zn_close = zn_raw["Adj Close"]
    else:
        raise RuntimeError(
            "Could not find ZN futures closing prices."
        )

# Handle a possible one-column DataFrame
if isinstance(zn_close, pd.DataFrame):
    zn_close = zn_close.iloc[:, 0]

zn_close.index = pd.to_datetime(zn_close.index)

# Monthly average futures price
zn_price = zn_close.resample("MS").mean()
zn_price.name = "ZN Futures Price"

# -----------------------------
# 4. Combine the monthly data
# -----------------------------
data = pd.concat(
    [
        pce_inflation,
        treasury_yield,
        zn_price,
    ],
    axis=1,
).sort_index()

# Keep available observations for each series separately.
# This avoids deleting a month from every graph just
# because one data source is missing an observation.
pce_plot = data["PCE Inflation (YoY)"].dropna()
yield_plot = data["10-Year Treasury Yield"].dropna()
zn_plot = data["ZN Futures Price"].dropna()

if pce_plot.empty or yield_plot.empty or zn_plot.empty:
    raise RuntimeError(
        "At least one series has no usable observations."
    )

# -----------------------------
# 5. Plot all three series
# -----------------------------
plt.style.use("seaborn-v0_8-whitegrid")

fig, axes = plt.subplots(
    3,
    1,
    figsize=(15, 11),
    sharex=True,
)

# Panel 1: PCE inflation
axes[0].plot(
    pce_plot.index,
    pce_plot,
    color="#2563eb",
    linewidth=2.2,
)

axes[0].set_title(
    "PCE Inflation (Year over Year)",
    fontsize=12,
    fontweight="bold",
)
axes[0].set_ylabel("Inflation (%)")
axes[0].grid(True, alpha=0.3)

# Panel 2: Treasury yield
axes[1].plot(
    yield_plot.index,
    yield_plot,
    color="#ea580c",
    linewidth=2.2,
)

axes[1].set_title(
    "10-Year US Treasury Yield",
    fontsize=12,
    fontweight="bold",
)
axes[1].set_ylabel("Yield (%)")
axes[1].grid(True, alpha=0.3)

# Panel 3: ZN futures price
axes[2].plot(
    zn_plot.index,
    zn_plot,
    color="#16a34a",
    linewidth=2.2,
)

axes[2].set_title(
    "ZN 10-Year Treasury Note Futures",
    fontsize=12,
    fontweight="bold",
)
axes[2].set_ylabel("Futures price (points)")
axes[2].set_xlabel("Date")
axes[2].grid(True, alpha=0.3)

fig.suptitle(
    "Inflation, Treasury Yields, and Treasury Futures",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Save data
# -----------------------------
data.to_csv("pce_yield_zn_futures.csv")

print("\nLatest monthly observations:")
print(data.tail(12).round(3))

print("\nSaved to pce_yield_zn_futures.csv")

# =============================================================================
# Build a (16 tickers) × (365 days) × (8 metrics) valuation tensor from Yahoo.
# =============================================================================

import numpy as np
import pandas as pd
import yfinance as yf

# ---- 1. Universe of tickers ------------------------------------------------
tickers = [
    'NVDA', 'AAPL', 'GOOG', 'GOOGL', 'MSFT',
    'AVGO', 'META', 'AMZN', 'TSLA', 'AMD',
    'ORCL', 'PLTR', 'IBM',  'TXN',  'QCOM', 'MU',
]

# ---- 2. Metric order -------------------------------------------------------
key_names = [
    "market_cap",   # 0 → price × shares outstanding
    "pe_ratio",     # 1 → price / TTM diluted EPS
    "forward_pe",   # 2 → snapshot-only (Yahoo has no forward-EPS history)
    "ps_ratio",     # 3 → market_cap / TTM revenue
    "pb_ratio",     # 4 → market_cap / stockholders' equity
    "peg_ratio",    # 5 → snapshot-only (needs analyst estimates history)
    "ev_ebitda",    # 6 → enterprise_value / TTM EBITDA
    "ev_revenue",   # 7 → enterprise_value / TTM revenue
]

# ---- 3. Date axis ----------------------------------------------------------
n_days = 365
end    = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
start  = end - pd.Timedelta(days=n_days)
dates  = pd.date_range(start=start, end=end - pd.Timedelta(days=1), freq="D")

# ---- 4. The tensor itself --------------------------------------------------
tensor = np.full((len(tickers), n_days, len(key_names)), np.nan, dtype=np.float64)


# =============================================================================
# Helper: align an arbitrary dated Series to our fixed `dates` calendar.
#
# CRITICAL FIX: get_shares_full() can return MULTIPLE rows per calendar day
# (intraday observations, or different timestamps that collapse to the same
# midnight after .normalize()). pandas `.reindex()` requires a UNIQUE source
# index — otherwise it raises "cannot reindex on an axis with duplicate
# labels". We deduplicate by keeping the LAST observation of each day,
# which is the most recent share count (the one that matters for market cap).
# =============================================================================
def _align(series, dates):
    if series is None or series.empty:
        return pd.Series(np.nan, index=dates, dtype=float)

    s = pd.to_numeric(series, errors="coerce").copy()

    # Strip tz (yfinance mixes tz-aware and tz-naive) and collapse to midnight.
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()

    # Sort, then drop duplicate days — keep the LAST row for each date.
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]     # ← the actual fix

    # ffill onto the union of observed days ∪ our 365-day window
    s = s.reindex(s.index.union(dates)).ffill()

    # Finally, cut down to exactly our `dates`
    return s.reindex(dates)


# =============================================================================
# Helper: pick a row from a quarterly financial-statement DataFrame.
# =============================================================================
def _quarterly(qdf, candidates, dates):
    if qdf is None or qdf.empty:
        return pd.Series(np.nan, index=dates)
    for name in candidates:
        if name in qdf.index:
            return _align(qdf.loc[name], dates)
    return pd.Series(np.nan, index=dates)


# =============================================================================
# Helper: trailing-4-quarter sum from a daily ffilled series.
# =============================================================================
def _ttm(daily_ffilled):
    return daily_ffilled.rolling(365, min_periods=1).mean() * 4


# =============================================================================
# Main loop
# =============================================================================
for i, sym in enumerate(tickers):
    try:
        tk = yf.Ticker(sym)

        # ---- 4a. Per-day price --------------------------------------------
        hist = tk.history(start=start, end=end, auto_adjust=False)
        if hist.empty:
            print(f"[skip] {sym}: no history")
            continue

        hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
        # Defensive: some tickers occasionally return duplicate dates.
        hist = hist[~hist.index.duplicated(keep="last")]
        price = hist["Close"].reindex(dates)

        # ---- 4b. Shares outstanding ---------------------------------------
        # Fetch `info` once up front — used here for sharesOutstanding and
        # again in 4e for forwardPE / pegRatio. One HTTP call, two uses.
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        try:
            shares_raw = tk.get_shares_full(start=start, end=end)
        except Exception:
            shares_raw = None

        # Per-day path — wrapped in its OWN try/except so that if _align
        # somehow still raises (bad data), we fall through to the snapshot
        # path instead of losing the entire ticker.
        shares     = None
        shares_src = "MISSING"
        if shares_raw is not None and not shares_raw.empty:
            try:
                shares     = _align(shares_raw, dates)
                shares_src = "per-day"
            except Exception:
                shares = None    # fall through to snapshot

        # Snapshot fallback — broadcast info["sharesOutstanding"] to 365 days.
        if shares is None:
            snap = info.get("sharesOutstanding")
            if snap is not None:
                shares     = pd.Series(float(snap), index=dates, dtype=float)
                shares_src = "snapshot"
            else:
                shares     = pd.Series(np.nan, index=dates, dtype=float)
                shares_src = "MISSING"

        # ---- 4c. Per-quarter fundamentals, ffilled to daily ---------------
        try: qbs = tk.quarterly_balance_sheet
        except Exception: qbs = None
        try: qis = tk.quarterly_income_stmt
        except Exception: qis = None

        equity  = _quarterly(qbs, ["Stockholders Equity",
                                   "Total Equity Gross Minority Interest"], dates)
        debt    = _quarterly(qbs, ["Total Debt"], dates).fillna(0)
        cash    = _quarterly(qbs, ["Cash And Cash Equivalents",
                                   "Cash Cash Equivalents And Short Term Investments"],
                             dates).fillna(0)
        revenue = _quarterly(qis, ["Total Revenue", "Operating Revenue"], dates)
        ebitda  = _quarterly(qis, ["EBITDA", "Normalized EBITDA"], dates)
        eps     = _quarterly(qis, ["Diluted EPS", "Basic EPS"], dates)

        ttm_rev    = _ttm(revenue)
        ttm_ebitda = _ttm(ebitda)
        ttm_eps    = _ttm(eps)

        # ---- 4d. Assemble the 8 columns -----------------------------------
        mcap     = price * shares
        net_debt = debt - cash
        ev       = mcap + net_debt

        tensor[i, :, 0] = mcap
        tensor[i, :, 1] = price / ttm_eps
        tensor[i, :, 3] = mcap / ttm_rev
        tensor[i, :, 4] = mcap / equity
        tensor[i, :, 6] = ev / ttm_ebitda
        tensor[i, :, 7] = ev / ttm_rev

        # ---- 4e. Snapshot-only metrics ------------------------------------
        fwd_pe = info.get("forwardPE")
        peg    = info.get("pegRatio")
        if fwd_pe is not None:
            tensor[i, :, 2] = price * 0 + fwd_pe
        if peg is not None:
            tensor[i, :, 5] = peg

        print(f"[ok] {sym}  shares={shares_src}")

    except Exception as e:
        # Print the exception TYPE + MESSAGE so the failure mode is obvious.
        print(f"[error] {sym}: {type(e).__name__}: {e}")


# ---- 5. Sanity output ------------------------------------------------------
print("tensor.shape =", tensor.shape)


# =============================================================================
# Health check
# =============================================================================
filled = ~np.isnan(tensor)
pct    = filled.mean(axis=1) * 100

df_fill = pd.DataFrame(
    pct.round(0).astype(int),    # lowercase "int" — pct is a numpy array
    index=tickers,
    columns=key_names,
)
print("\n% of 365 days populated (0 = metric empty for that ticker):")
print(df_fill.to_string())

empty = [(t, m) for i, t in enumerate(tickers)
                for k, m in enumerate(key_names)
                if np.isnan(tensor[i, :, k]).all()]
if empty:
    print("\nFully-empty (ticker, metric) pairs:")
    for t, m in empty:
        print(f"  {t:6s} {m}")
else:
    print("\nNo fully-empty columns — nice.")
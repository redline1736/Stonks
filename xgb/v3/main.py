"""
Stonks-XGB v3.0 — Expanded universe + richer features
====================================================
- 23 sectors, 200+ stocks (20 sectors × 10+ stocks)
- 20+ ETFs (sector ETFs, country ETFs, commodities, bonds, volatility)
- 5-day forward target (top-30% cross-sectional)
- Walk-forward CV (folds 2020-22, 2022-24; test 2025+)
- Heavier XGBoost: max_depth 6, lr 0.02, 6000 rounds
- Richer feature set:
  * Technical: returns (1,2,3,5,10,21,63,120,252), momentum_12_1,
    SMA/EMA ratios, MACD, RSI, BB, ATR, OBV, ADX, Williams %R, CCI, MFI,
    volatility, drawdown, gap, range, body
  * Cross-sectional: rank, z-score, sector-relative for all key features
  * Regime: SPY trend, VIX proxy (^VIX), 10Y yield (^TNX), USD (DX-Y.NYB)
  * Fundamentals: PE, forward PE, PEG, PS, PB, EV/EBITDA, rev/eps growth,
    margins (gross/op/net), ROE, ROA, current/quick ratio, debt/equity,
    FCF yield, dividend yield, payout ratio, insider/institutional ownership,
    short interest, beta, 52w high/low position
- 5-seed ensemble, rate-targeted threshold (20% trade rate, test-ensemble calibrated)
- RobustScaler, high-NaN drop, corr drop (>0.95), 1-99% clip
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import pickle
import json
import time
import glob
import warnings
import argparse
import joblib
from datetime import datetime, timedelta
from tqdm import tqdm

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score)
import xgboost as xgb

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG — UNIVERSE
# =========================================================

SECTOR_TICKERS = {
    "Information Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "CRM", "AMD", "INTC", "CSCO",
        "ORCL", "IBM", "TXN", "QCOM", "MU", "ADI", "LRCX", "KLAC",
        "PANW", "CRWD", "DDOG", "MDB", "ZS", "NET", "SMCI", "PLTR",
        "SNOW", "ADBE", "INTU", "NOW", "WDAY",
    ],
    "Communication Services": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "SNAP",
        "TMUS", "CHTR", "EA", "TTWO", "WBD", "PARA", "OMC", "IPG",
        "MTCH", "PINS", "RBLX", "SPOT",
    ],
    "Consumer Cyclical": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG",
        "LULU", "CMG", "CROX", "DECK", "ANF", "AEO", "RL", "ROKU",
        "UBER", "SHOP", "DKNG", "MAR", "HLT", "F", "GM", "STLA",
        "RIVN", "LCID", "NCLH", "CCL", "RCL", "EXPE",
    ],
    "Financials": [
        "JPM", "V", "MA", "BAC", "WFC", "BLK", "GS", "MS", "AXP", "C",
        "SCHW", "KKR", "APO", "ARES", "USB", "PNC", "TFC", "COF",
        "BX", "STT", "ICE", "CME", "MMC", "AON", "MET", "PRU", "AFL",
        "TRV", "ALL", "HIG",
    ],
    "Healthcare": [
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "TMO", "DHR",
        "VRTX", "REGN", "MRNA", "ABT", "MDT", "SYK", "BSX", "EW",
        "ISRG", "BDX", "CI", "ELV", "HCA", "HUM", "CNC", "MOH",
        "ZTS", "IDXX", "DXCM", "IQV", "STE", "RMD",
    ],
    "Consumer Defensive": [
        "WMT", "PG", "COST", "KO", "PEP", "CL", "MDLZ", "MO",
        "PM", "UL", "NSRGY", "KDP", "MNST", "KHC", "GIS", "K",
        "HSY", "HRL", "TSN", "CAG", "CPB", "SJM", "CHD", "CLX",
        "KVUE", "EL", "KMB",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "VLO",
        "PSX", "PXD", "DVN", "HAL", "BKR", "WMB", "OKE", "CTRA",
        "FANG", "MRO", "APA", "HES", "EQT", "AR", "PR", "RRC",
    ],
    "Industrials": [
        "CAT", "GE", "BA", "UPS", "RTX", "HON", "MMM", "LMT",
        "DE", "SPGI", "UNP", "CSX", "NSC", "FDX", "EMR", "ETN",
        "PH", "CMI", "PCAR", "JCI", "TT", "ITW", "WM", "RSG",
        "URI", "FAST", "ODFL", "VRSK", "IR", "AME", "ROP",
    ],
    "Materials": [
        "LIN", "SHW", "APD", "ECL", "NEM", "FCX", "DOW", "DD",
        "PPG", "LYB", "ALB", "CE", "EMN", "VMC", "MLM", "IFF",
        "FMC", "CF", "MOS", "NUE", "STLD", "PKG", "IP", "WRK",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "CEG", "AEP", "SRE", "EXC", "XEL",
        "PEG", "WEC", "ED", "EIX", "D", "ETR", "ES", "AWK",
        "ATO", "CMS", "DTE", "FE", "AEE", "LNT", "NI", "PPL",
    ],
    "Real Estate": [
        "PLD", "AMT", "EQIX", "WELL", "SPG", "PSA", "O", "DLR",
        "VICI", "AVB", "EQR", "ESS", "MAA", "UDR", "CPT", "HST",
        "BXP", "KIM", "REG", "FRT", "ARE", "BAM", "EXR", "IRM",
    ],
    "Data Centers & Infra": [
        "VRT", "STRL", "IRM", "NVT", "ENPH", "FSLR", "EQT",
        "TT", "ANET", "CCI", "AMT", "EQIX", "DLR", "SBAC",
    ],
    "Semiconductors": [
        "MRVL", "ON", "MPWR", "ENPH", "SWKS", "QRVO", "MCHP",
        "NXPI", "WOLF", "ARM", "CRDO", "LSCC", "ALGM", "POWI",
    ],
    "Aerospace & Defense": [
        "TXT", "HII", "LDOS", "HEI", "TDG", "LHX", "NOC", "GD",
        "BA", "RTX", "LMT", "HWM", "AXON", "RKLB", "SPCE",
    ],
    "Biotech": [
        "BIIB", "GILD", "AMGN", "BMRN", "ALNY", "INCY", "EXEL",
        "UTHR", "IONS", "RARE", "BLUE", "CRSP", "NTLA", "BEAM",
        "VRTX", "REGN", "ARGX", "BBIO", "FOLD", "PTCT",
    ],
    "REITs Specialized": [
        "VICI", "AVB", "PSA", "EXR", "CUBE", "LSI", "NHI",
        "DOC", "WPC", "NNN", "STOR", "GTY", "EPRT", "BXMT",
    ],
    "Fintech": [
        "SQ", "PYPL", "COIN", "HOOD", "AFRM", "SOFI", "UPST",
        "LC", "NU", "MELI", "DASH", "GLXY", "RIOT", "MARA",
    ],
    "Cybersecurity": [
        "CRWD", "PANW", "ZS", "DDOG", "S", "OKTA", "FTNT",
        "NET", "CYBR", "RPD", "SPLK", "TENB", "QLYS", "VRNS",
    ],
    "E-Commerce": [
        "AMZN", "SHOP", "MELI", "EBAY", "ETSY", "W", "CHWY",
        "RVLV", "GLBE", "SE", "BABA", "JD", "PDD", "MNSO",
    ],
    "Cloud Software": [
        "MSFT", "CRM", "NOW", "SNOW", "DDOG", "MDB", "NET",
        "TEAM", "ATLASSIAN", "ADBE", "WDAY", "ORCL", "INTU", "VEEV",
    ],
    "EV & Clean Energy": [
        "TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI", "F", "GM",
        "ENPH", "FSLR", "SEDG", "RUN", "JKS", "ARRY", "NEE",
    ],
    "Asset Management": [
        "BLK", "BX", "KKR", "APO", "ARES", "AMP", "TROW", "IVZ",
        "BEN", "FHI", "JHG", "AB", "LM", "VCTR", "BSIG",
    ],
    "Insurance": [
        "BRK-B", "AIG", "PRU", "MET", "AFL", "TRV", "ALL", "HIG",
        "WRB", "GL", "LNC", "PFG", "CNA", "RGA", "MTG", "FNF",
    ],
}

# Expanded ETF universe — sector, country, commodity, bond, vol
ETF_TICKERS = [
    # Broad market
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
    # Sector ETFs
    "XLF", "XLE", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB",
    "XLC", "XLRE",
    # Country / international
    "EFA", "EEM", "VEA", "VWO", "EWJ", "EWZ", "MCHI", "INDA",
    # Bonds
    "TLT", "IEF", "SHY", "BIL", "LQD", "HYG", "JNK", "MUB", "TIP",
    # Commodities
    "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PPLT",
    # REITs
    "VNQ", "SCHH", "USRT", "REM",
    # Volatility
    "VXX", "UVXY", "SVXY",
    # Crypto
    "BITO", "IBIT", "ETHE",
    # Thematic
    "ARKK", "ARKW", "ARKF", "SOXL", "TQQQ", "SQQQ", "SPHB", "SPLV",
]
ETF_SECTOR_NAME = "ETF_Macro"

TICKER_SECTOR = {}
for sector, tickers in SECTOR_TICKERS.items():
    for t in tickers:
        TICKER_SECTOR[t] = sector
for etf in ETF_TICKERS:
    TICKER_SECTOR[etf] = ETF_SECTOR_NAME

TICKERS = list(TICKER_SECTOR.keys())
SECTOR_NAMES = sorted(set(TICKER_SECTOR.values()))

# External market data tickers (for regime features)
REGIME_TICKERS = ["^VIX", "^TNX", "DX-Y.NYB", "^FVX", "CL=F", "GC=F"]

TRAIN_START = "2000-01-01"
TEST_END    = datetime.now().strftime("%Y-%m-%d")

INTERVAL = "1d"
FUTURE_DAYS = 5
EPS = 1e-9

TARGET_MODE = "top_30"   # top 30% of 5-day returns

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CACHE_FILE    = os.path.join(MODEL_DIR, "stonks_data_v3.pkl")
FUNDA_CACHE   = os.path.join(MODEL_DIR, "fundamentals_v3.pkl")
REGIME_CACHE  = os.path.join(MODEL_DIR, "regime_v3.pkl")
MODEL_PATH    = os.path.join(MODEL_DIR, "xgboost_model_v3.json")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_list_v3.json")
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler_v3.pkl")
THRESH_PATH   = os.path.join(MODEL_DIR, "threshold_v3.json")

# Training knobs
NUM_BOOST_ROUND      = 6000
EARLY_STOPPING_ROUND = 200
LEARNING_RATE        = 0.02
FINAL_ROUNDS_MULT    = 1.25
TARGET_TRADE_RATE    = 0.20

# Lookback for predict
PREDICT_LOOKBACK_DAYS = 700

print(f"Total stocks: {len(TICKERS)} across {len(SECTOR_NAMES)} sectors")
print(f"  - {len(SECTOR_TICKERS)} sector groups, "
      f"{len(ETF_TICKERS)} ETFs, {len(REGIME_TICKERS)} regime tickers")
print(f"Daily data from {TRAIN_START} to {TEST_END}")
print(f"Target: top {int((1-0.30)*100)}% of {FUTURE_DAYS}d returns")
print(f"Training: num_boost_round={NUM_BOOST_ROUND}, "
      f"early_stopping={EARLY_STOPPING_ROUND}, lr={LEARNING_RATE}")


# =========================================================
# REGIME / MACRO DATA (VIX, TNX, DXY, commodities, bonds)
# =========================================================

def fetch_regime_data():
    """Fetch and cache external market regime data."""
    if os.path.exists(REGIME_CACHE):
        print("Loading cached regime data...")
        with open(REGIME_CACHE, "rb") as f:
            return pickle.load(f)

    print("Fetching regime data...")
    regime = {}
    for ticker in REGIME_TICKERS:
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=TRAIN_START, end=TEST_END,
                           interval=INTERVAL, auto_adjust=True)
            if df.empty:
                continue
            keep = [c for c in ["Close", "Volume"] if c in df.columns]
            df = df[keep].copy()
            df.index.name = "Date"
            df = df.reset_index()
            # Clean ticker name for column prefix
            clean = ticker.replace("^", "").replace("-", "_").replace("=", "_")
            df = df.rename(columns={"Close": f"{clean}_Close",
                                    "Volume": f"{clean}_Vol"})
            regime[ticker] = df
            print(f"  {ticker}: {len(df)} days")
            time.sleep(0.3)
        except Exception as e:
            print(f"  FAILED {ticker}: {e}")

    with open(REGIME_CACHE, "wb") as f:
        pickle.dump(regime, f)
    return regime


def add_regime_features(df, regime):
    """Merge regime features into the main dataframe."""
    df = df.copy()
    for ticker, rdf in regime.items():
        clean = ticker.replace("^", "").replace("-", "_").replace("=", "_")
        close_col = f"{clean}_Close"
        if close_col not in rdf.columns:
            continue
        # Compute returns
        rdf[f"{clean}_Return_1d"] = rdf[close_col].pct_change(1)
        rdf[f"{clean}_Return_5d"] = rdf[close_col].pct_change(5)
        rdf[f"{clean}_Return_21d"] = rdf[close_col].pct_change(21)
        # Z-score regime
        rdf[f"{clean}_Z_60d"] = (
            (rdf[close_col] - rdf[close_col].rolling(60).mean()) /
            (rdf[close_col].rolling(60).std() + EPS)
        )
        # Merge
        merge_cols = ["Date", f"{clean}_Return_1d", f"{clean}_Return_5d",
                      f"{clean}_Return_21d", f"{clean}_Z_60d"]
        merge_cols = [c for c in merge_cols if c in rdf.columns]
        rdf_sub = rdf[merge_cols].drop_duplicates("Date")
        df = df.merge(rdf_sub, on="Date", how="left")
    return df


# =========================================================
# FUNDAMENTALS (expanded with insider/inst ownership, short int, beta)
# =========================================================

def fetch_all_fundamentals():
    if os.path.exists(FUNDA_CACHE):
        print("Loading cached fundamentals...")
        with open(FUNDA_CACHE, "rb") as f:
            return pickle.load(f)
    print("Fetching fundamentals...")
    funda = {}
    for ticker in tqdm(TICKERS, desc="Fundamentals"):
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            funda[ticker] = {
                # Valuation
                "market_cap":     info.get("marketCap", np.nan),
                "pe_ratio":       info.get("trailingPE", np.nan),
                "forward_pe":     info.get("forwardPE", np.nan),
                "ps_ratio":       info.get("priceToSalesTrailing12Months", np.nan),
                "pb_ratio":       info.get("priceToBook", np.nan),
                "peg_ratio":      info.get("pegRatio", np.nan),
                "ev_ebitda":      info.get("enterpriseToEbitda", np.nan),
                "ev_revenue":     info.get("enterpriseToRevenue", np.nan),
                # Growth
                "rev_growth":     info.get("revenueGrowth", np.nan),
                "eps_growth":     info.get("earningsGrowth", np.nan),
                # Profitability
                "current_ratio":  info.get("currentRatio", np.nan),
                "quick_ratio":    info.get("quickRatio", np.nan),
                "debt_equity":    info.get("debtToEquity", np.nan),
                "op_margin":      info.get("operatingMargins", np.nan),
                "net_margin":     info.get("profitMargins", np.nan),
                "gross_margin":   info.get("grossMargins", np.nan),
                "roe":            info.get("returnOnEquity", np.nan),
                "roa":            info.get("returnOnAssets", np.nan),
                # Cash flow
                "fcf_yield":      info.get("freeCashFlowYield", np.nan),
                "ocf_yield":      (info.get("operatingCashflow", 0) /
                                    max(info.get("marketCap", 1), 1)
                                    if info.get("operatingCashflow") else np.nan),
                # Capital return
                "div_yield":      info.get("dividendYield", np.nan),
                "payout_ratio":   info.get("payoutRatio", np.nan),
                # Ownership
                "insider_pct":    info.get("heldPercentInsiders", np.nan),
                "inst_pct":       info.get("heldPercentInstitutions", np.nan),
                "short_pct":      info.get("shortPercentOfFloat", np.nan),
                # Risk
                "beta":           info.get("beta", np.nan),
                "52w_high":       info.get("fiftyTwoWeekHigh", np.nan),
                "52w_low":        info.get("fiftyTwoWeekLow", np.nan),
                "52w_change":     info.get("52WeekChange", np.nan),
                # Size category
                "shares_out":     info.get("sharesOutstanding", np.nan),
                "float_shares":   info.get("floatShares", np.nan),
            }
            time.sleep(0.15)
        except:
            funda[ticker] = {}
    with open(FUNDA_CACHE, "wb") as f:
        pickle.dump(funda, f)
    return funda


# =========================================================
# DOWNLOAD
# =========================================================

def fetch_and_cache():
    if os.path.exists(CACHE_FILE):
        print("Loading cached daily data (v3)...")
        with open(CACHE_FILE, "rb") as f:
            df = pickle.load(f)
        print(f"Loaded {len(df):,} rows across {df['Ticker'].nunique()} tickers")
        return df

    os.makedirs(os.path.join(MODEL_DIR, "partial_daily_v3"), exist_ok=True)
    all_dfs = []
    for ticker in tqdm(TICKERS, desc="Download"):
        cached_path = os.path.join(MODEL_DIR, "partial_daily_v3", f"{ticker}.pkl")
        if os.path.exists(cached_path):
            all_dfs.append(pd.read_pickle(cached_path))
            continue
        try:
            t = yf.Ticker(ticker)
            df_t = t.history(start=TRAIN_START, end=TEST_END,
                              interval=INTERVAL, auto_adjust=True)
            if df_t.empty:
                continue
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"]
                    if c in df_t.columns]
            df_t = df_t[keep].copy()
            df_t["Ticker"] = ticker
            df_t.index.name = "Date"
            df_t = df_t.reset_index()
            df_t.to_pickle(cached_path)
            all_dfs.append(df_t)
            time.sleep(0.2)
        except Exception as e:
            print(f"  FAILED {ticker}: {e}")

    df = pd.concat(all_dfs, ignore_index=False)
    df.to_pickle(CACHE_FILE)
    print(f"Downloaded {len(df):,} rows, {df['Ticker'].nunique()} tickers")
    return df


# =========================================================
# PER-TICKER FEATURES (expanded)
# =========================================================

def add_features(df_ticker):
    df_ticker = df_ticker.sort_values("Date").copy()
    close  = df_ticker["Close"].astype(float)
    high   = df_ticker["High"].astype(float)
    low    = df_ticker["Low"].astype(float)
    volume = df_ticker["Volume"].astype(float)
    open_p = df_ticker["Open"].astype(float)

    # --- Returns (short, medium, long) ---
    for p in [1, 2, 3, 5, 10, 21, 63, 120, 252]:
        df_ticker[f"Return_{p}d"] = close.pct_change(p)

    df_ticker["Log_Return"] = np.log((close + EPS) / (close.shift(1) + EPS))

    # --- Momentum factors (academic) ---
    df_ticker["Momentum_12_1"] = close.pct_change(252) - close.pct_change(21)
    df_ticker["Momentum_6_1"]  = close.pct_change(126) - close.pct_change(21)

    # --- Moving averages ---
    for p in [5, 10, 20, 50, 100, 200]:
        df_ticker[f"SMA_{p}"]       = close.rolling(p).mean()
        df_ticker[f"SMA_{p}_Ratio"] = close / (df_ticker[f"SMA_{p}"] + EPS)
        df_ticker[f"EMA_{p}"]       = close.ewm(span=p, adjust=False).mean()
        df_ticker[f"EMA_{p}_Ratio"] = close / (df_ticker[f"EMA_{p}"] + EPS)

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df_ticker["MACD"]        = ema12 - ema26
    df_ticker["MACD_Signal"] = df_ticker["MACD"].ewm(span=9, adjust=False).mean()
    df_ticker["MACD_Hist"]   = df_ticker["MACD"] - df_ticker["MACD_Signal"]
    df_ticker["MACD_Cross"]  = (df_ticker["MACD"] > df_ticker["MACD_Signal"]).astype(float)
    df_ticker["MACD_Accel"]  = df_ticker["MACD_Hist"].diff(3)

    # --- RSI ---
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs    = gain / (loss + EPS)
    df_ticker["RSI"]        = 100 - (100 / (1 + rs))
    df_ticker["RSI_Change"] = df_ticker["RSI"].diff(5)
    df_ticker["RSI_MA"]     = df_ticker["RSI"].rolling(5).mean()

    # --- Bollinger Bands ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df_ticker["BB_Width"]    = (2 * bb_std) / (bb_mid + EPS)
    df_ticker["BB_Position"] = (close - bb_mid) / (bb_std + EPS)
    df_ticker["BB_%B"]       = (close - (bb_mid - 2*bb_std)) / (4*bb_std + EPS)

    # --- ATR ---
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    df_ticker["ATR"]       = tr.rolling(14).mean()
    df_ticker["ATR_Ratio"] = df_ticker["ATR"] / (close + EPS)

    # --- Volume + OBV ---
    df_ticker["Volume_SMA_20"]    = volume.rolling(20).mean()
    df_ticker["Volume_Ratio"]     = volume / (df_ticker["Volume_SMA_20"] + EPS)
    df_ticker["Volume_Change_5d"] = volume.pct_change(5)

    # OBV (On-Balance Volume)
    direction = np.sign(close.diff()).fillna(0)
    df_ticker["OBV"]       = (direction * volume).cumsum()
    df_ticker["OBV_SMA_20"]= df_ticker["OBV"].rolling(20).mean()
    df_ticker["OBV_Ratio"] = df_ticker["OBV"] / (df_ticker["OBV_SMA_20"] + EPS)

    # --- Volatility ---
    df_ticker["Volatility_5d"]    = df_ticker["Log_Return"].rolling(5).std()
    df_ticker["Volatility_20d"]   = df_ticker["Log_Return"].rolling(20).std()
    df_ticker["Volatility_60d"]   = df_ticker["Log_Return"].rolling(60).std()
    df_ticker["Volatility_Ratio"] = df_ticker["Volatility_20d"] / (df_ticker["Volatility_60d"] + EPS)

    # --- Drawdown ---
    rolling_max = close.cummax()
    df_ticker["Drawdown"]     = (close - rolling_max) / (rolling_max + EPS)
    df_ticker["Drawdown_60d"] = close / (close.rolling(60).max() + EPS) - 1

    # --- Price position in range ---
    for p in [20, 50, 200]:
        hp = high.rolling(p).max()
        lp = low.rolling(p).min()
        df_ticker[f"Price_Position_{p}"] = (close - lp) / (hp - lp + EPS)

    # --- Gap ---
    df_ticker["Gap"]      = (open_p - close.shift(1)) / (close.shift(1) + EPS)
    df_ticker["Gap_5d_Avg"] = df_ticker["Gap"].rolling(5).mean()

    # --- Candle structure ---
    df_ticker["Body"]       = close - open_p
    df_ticker["Upper_Wick"] = high - np.maximum(close, open_p)
    df_ticker["Lower_Wick"] = np.minimum(close, open_p) - low
    df_ticker["Range"]      = high - low
    df_ticker["Body_Ratio"] = abs(df_ticker["Body"]) / (df_ticker["Range"] + EPS)
    df_ticker["Is_Green"]   = (close > open_p).astype(float)

    # --- Williams %R (14) ---
    hh = high.rolling(14).max()
    ll = low.rolling(14).min()
    df_ticker["Williams_R"] = (hh - close) / (hh - ll + EPS)

    # --- CCI (Commodity Channel Index, 20) ---
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df_ticker["CCI"] = (tp - sma_tp) / (0.015 * mad + EPS)

    # --- MFI (Money Flow Index, 14) ---
    mf = tp * volume
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    mf_ratio = pos_mf / (neg_mf + EPS)
    df_ticker["MFI"] = 100 - (100 / (1 + mf_ratio))

    # --- ADX (Average Directional Index, 14) — simplified ---
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm  = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr14 = tr.rolling(14).mean()
    plus_di  = 100 * (plus_dm.rolling(14).mean() / (tr14 + EPS))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (tr14 + EPS))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    df_ticker["ADX"]       = dx.rolling(14).mean()
    df_ticker["Plus_DI"]   = plus_di
    df_ticker["Minus_DI"]  = minus_di

    # --- Future return (target) ---
    df_ticker["Future_Close"]  = close.shift(-FUTURE_DAYS)
    df_ticker["Future_Return"] = (df_ticker["Future_Close"] - close) / (close + EPS)

    # NaN guard
    for col in df_ticker.select_dtypes(include=[np.number]).columns:
        df_ticker[col] = df_ticker[col].replace([np.inf, -np.inf], np.nan)

    return df_ticker


# =========================================================
# CROSS-SECTIONAL + MARKET REGIME FEATURES
# =========================================================

def add_cross_sectional_and_regime(df, regime):
    df = df.copy()
    df["Sector"] = df["Ticker"].map(TICKER_SECTOR)

    # --- SPY (core market regime) ---
    spy_cols = [
        "Date", "Close", "Volume",
        "Return_1d", "Return_2d", "Return_3d", "Return_5d",
        "Return_10d", "Return_21d", "Return_63d", "Return_120d", "Return_252d",
        "Momentum_12_1", "Momentum_6_1",
        "Volatility_5d", "Volatility_20d", "Volatility_60d",
        "SMA_20", "SMA_50", "SMA_100", "SMA_200",
        "SMA_20_Ratio", "SMA_50_Ratio", "SMA_200_Ratio",
        "RSI", "ATR_Ratio", "BB_Position", "ADX", "MFI", "Williams_R",
    ]
    spy_cols = [c for c in spy_cols if c in df.columns]
    spy = df[df["Ticker"] == "SPY"][spy_cols].copy()
    spy = spy.sort_values("Date").reset_index(drop=True)

    for p in [20, 50, 100, 200]:
        spy[f"SPY_SMA_{p}"]   = spy["Close"].rolling(p).mean()
        spy[f"SPY_Above_{p}"] = (spy["Close"] > spy[f"SPY_SMA_{p}"]).astype(float)
        spy[f"SPY_Slope_{p}"] = spy[f"SPY_SMA_{p}"].pct_change(5)

    spy["SPY_Trend_Strength"] = (
        spy["SPY_Above_20"] + spy["SPY_Above_50"] +
        spy["SPY_Above_100"] + spy["SPY_Above_200"]
    ) / 4.0
    spy["SPY_Vol_Regime"] = spy["Volatility_20d"].rolling(60).rank(pct=True)
    spy["SPY_Drawdown"]   = spy["Close"] / spy["Close"].cummax() - 1

    # Market breadth
    if "SMA_50" in df.columns:
        breadth = (
            df.assign(above_50=(df["Close"] > df["SMA_50"]).astype(float))
              .groupby("Date")["above_50"].mean()
              .rename("Market_Breadth_50")
              .reset_index()
        )
        breadth["Market_Breadth_200"] = (
            df.assign(above_200=(df["Close"] > df["SMA_200"]).astype(float))
              .groupby("Date")["above_200"].mean()
              .values
        )
    else:
        breadth = pd.DataFrame({"Date": df["Date"].unique(),
                                 "Market_Breadth_50": 0.5,
                                 "Market_Breadth_200": 0.5})
    spy = spy.merge(breadth, on="Date", how="left")

    regime_cols = [c for c in spy.columns
                   if c.startswith("SPY_") or c.startswith("Market_Breadth_")]
    df = df.merge(spy[["Date"] + regime_cols], on="Date", how="left")

    # --- Cross-sectional ranks & z-scores ---
    cs_cols = [
        "Return_1d", "Return_5d", "Return_10d", "Return_21d", "Return_63d",
        "Return_120d", "Return_252d",
        "Momentum_12_1", "Momentum_6_1",
        "RSI", "RSI_Change",
        "Volatility_5d", "Volatility_20d", "ATR_Ratio",
        "Volume_Ratio", "Volume_Change_5d", "OBV_Ratio",
        "SMA_20_Ratio", "SMA_50_Ratio", "SMA_200_Ratio",
        "Price_Position_20", "Price_Position_50", "Price_Position_200",
        "BB_Position", "BB_%B",
        "MACD_Hist", "Williams_R", "CCI", "MFI", "ADX",
    ]
    cs_cols = [c for c in cs_cols if c in df.columns]

    for col in cs_cols:
        df[f"{col}_Rank"] = df.groupby("Date")[col].rank(pct=True)
        g = df.groupby("Date")[col]
        df[f"{col}_Z"] = (df[col] - g.transform("mean")) / (g.transform("std") + EPS)

    # Sector-relative
    for col in ["Return_5d", "Return_10d", "Return_21d", "Return_63d",
                "Return_120d", "Momentum_12_1", "Momentum_6_1",
                "RSI", "Volatility_20d", "Volume_Ratio", "BB_Position", "ADX"]:
        if col in df.columns:
            sec_mean = df.groupby(["Date", "Sector"])[col].transform("mean")
            df[f"{col}_SectorRel"] = df[col] - sec_mean

    # vs SPY
    return_cols = ["Return_1d", "Return_5d", "Return_10d", "Return_21d", "Return_63d"]
    for col in return_cols:
        if col in df.columns and col in spy.columns:
            spy_map = spy.set_index("Date")[col]
            df[f"{col}_vsSPY"] = df[col] - df["Date"].map(spy_map)

    # --- External regime (VIX, TNX, DXY, commodities) ---
    if regime:
        df = add_regime_features(df, regime)

    return df


# =========================================================
# BUILD DATASET
# =========================================================

def build_dataset():
    print("=" * 60)
    print(f"BUILDING DATASET v3 — FUTURE_DAYS={FUTURE_DAYS}")
    print("=" * 60)

    df = fetch_and_cache()
    if "Date" not in df.columns:
        df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Sector"] = df["Ticker"].map(TICKER_SECTOR)

    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(np.float64)
        elif df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    funda = fetch_all_fundamentals()
    regime = fetch_regime_data()

    print("Engineering per-ticker features...")
    processed = []
    dropped = []
    for ticker in tqdm(TICKERS, desc="Features"):
        tdf = df[df["Ticker"] == ticker].copy()
        if len(tdf) < 300:
            dropped.append(ticker)
            continue
        tdf = add_features(tdf)
        processed.append(tdf)
    if dropped:
        print(f"Dropped (insufficient history <300d): {dropped}")
    df = pd.concat(processed)
    df = df.replace([np.inf, -np.inf], np.nan)

    print("Adding cross-sectional + regime features...")
    df = add_cross_sectional_and_regime(df, regime)

    print("Adding fundamentals...")
    funda_cols = [
        "market_cap", "pe_ratio", "forward_pe", "ps_ratio", "pb_ratio",
        "peg_ratio", "ev_ebitda", "ev_revenue",
        "rev_growth", "eps_growth",
        "current_ratio", "quick_ratio", "debt_equity",
        "op_margin", "net_margin", "gross_margin", "roe", "roa",
        "fcf_yield", "ocf_yield", "div_yield", "payout_ratio",
        "insider_pct", "inst_pct", "short_pct", "beta",
        "52w_high", "52w_low", "52w_change", "shares_out", "float_shares",
    ]
    for ticker in TICKERS:
        mask = df["Ticker"] == ticker
        if mask.sum() == 0:
            continue
        f = funda.get(ticker, {})
        for col in funda_cols:
            df.loc[mask, col] = f.get(col, np.nan)

    # Target
    if TARGET_MODE == "top_30":
        df["Target"] = (
            df.groupby("Date")["Future_Return"]
              .transform(lambda x: (x >= x.quantile(0.70)).astype(int))
        )
    elif TARGET_MODE == "top_20":
        df["Target"] = (
            df.groupby("Date")["Future_Return"]
              .transform(lambda x: (x >= x.quantile(0.80)).astype(int))
        )
    elif TARGET_MODE == "beat_spy":
        spy = df[df["Ticker"] == "SPY"][["Date", f"Return_{FUTURE_DAYS}d"]].rename(
            columns={f"Return_{FUTURE_DAYS}d": f"SPY_Return_{FUTURE_DAYS}d"})
        df = df.merge(spy, on="Date", how="left")
        df["Target"] = ((df["Future_Return"] - df[f"SPY_Return_{FUTURE_DAYS}d"]) > 0).astype(int)
    else:
        raise ValueError(f"Unknown TARGET_MODE: {TARGET_MODE}")

    print(f"Target: {df['Target'].value_counts().to_dict()}")
    print(f"Avg Future Return ({FUTURE_DAYS}d): {df['Future_Return'].mean():.4f}")

    # Feature selection
    feature_keywords = [
        "Return_", "Log_Return", "SMA_", "EMA_", "MACD", "RSI",
        "BB_", "ATR", "Volume_", "OBV", "Volatility", "Price_Position",
        "Body", "Wick", "Range", "Is_Green", "Body_Ratio",
        "Gap", "Drawdown", "Williams_R", "CCI", "MFI", "ADX", "Plus_DI", "Minus_DI",
        "Momentum_12_1", "Momentum_6_1",
        "market_cap", "pe_ratio", "forward_pe", "ps_ratio", "pb_ratio",
        "peg_ratio", "ev_ebitda", "ev_revenue",
        "rev_growth", "eps_growth", "current_ratio", "quick_ratio",
        "debt_equity", "op_margin", "net_margin", "gross_margin",
        "roe", "roa", "fcf_yield", "ocf_yield", "div_yield", "payout_ratio",
        "insider_pct", "inst_pct", "short_pct", "beta",
        "52w_high", "52w_low", "52w_change", "shares_out", "float_shares",
        "_Rank", "_Z", "_SectorRel", "_vsSPY",
        "SPY_", "Market_Breadth_",
        "VIX_", "TNX_", "DX_Y_NYB_", "FVX_", "CL_F_", "GC_F_",
    ]
    exclude = {"Date", "Ticker", "Sector", "Future_Close", "Future_Return",
               "Target", f"SPY_Return_{FUTURE_DAYS}d"}
    feature_columns = sorted(set(
        c for c in df.columns
        if any(k in c for k in feature_keywords) and c not in exclude
    ))

    # Sector one-hot
    sd = pd.get_dummies(df["Sector"], prefix="Sector")
    df = pd.concat([df, sd], axis=1)
    df["Is_ETF"] = df["Ticker"].isin(ETF_TICKERS).astype(np.float64)

    for c in df.columns:
        if c.startswith("Sector_") or c == "Is_ETF":
            if df[c].dtype == bool:
                df[c] = df[c].astype(np.float64)

    sector_cols = [c for c in df.columns if c.startswith("Sector_")]
    feature_columns = sorted(set(feature_columns + sector_cols + ["Is_ETF"]))

    # Drop high-NaN
    nan_c = df[feature_columns].isna().sum()
    bad = nan_c[nan_c > len(df) * 0.5].index.tolist()
    if bad:
        print(f"Removing {len(bad)} high-NaN features")
        feature_columns = [c for c in feature_columns if c not in bad]

    # Coerce + fill
    for col in feature_columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].dtype == bool:
            df[col] = df[col].astype(np.float64)

    for col in feature_columns:
        if col not in df.columns:
            continue
        s = df[col]
        if s.isna().all():
            df[col] = 0.0
        else:
            med = s.median()
            df[col] = s.fillna(0 if pd.isna(med) else med).astype(np.float64)

    # Drop correlated
    corr = df[feature_columns].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    if to_drop:
        print(f"Removing {len(to_drop)} highly correlated features")
        feature_columns = [c for c in feature_columns if c not in to_drop]

    # Clip
    for col in feature_columns:
        if col not in df.columns:
            continue
        s = df[col]
        if s.std() == 0 or s.nunique() < 2:
            continue
        try:
            q1, q99 = s.quantile(0.01), s.quantile(0.99)
            df[col] = s.clip(q1, q99).astype(np.float64)
        except Exception as e:
            print(f"  Clip skipped for {col}: {e}")
            df[col] = s.astype(np.float64)

    before = len(df)
    df = df.dropna(subset=["Target"])
    print(f"After drop: {len(df):,} rows (dropped {before - len(df):,})")
    print(f"Total features: {len(feature_columns)}")

    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_columns, f)

    return df, feature_columns


# =========================================================
# THRESHOLD TUNING
# =========================================================

def find_optimal_threshold(y_true, y_proba, metric="f1"):
    best_t, best_v = 0.5, -1
    for t in np.arange(0.10, 0.90, 0.005):
        pred = (y_proba > t).astype(int)
        if metric == "accuracy":
            v = (pred == y_true).mean()
        else:
            v = f1_score(y_true, pred, zero_division=0)
        if v > best_v:
            best_v, best_t = v, t
    return best_t, best_v


# =========================================================
# TRAINING v3
# =========================================================

def train_xgboost(df, feature_columns):
    print("\n" + "=" * 60)
    print(f"TRAINING XGBOOST v3 — TARGET={TARGET_MODE}, "
          f"FUTURE_DAYS={FUTURE_DAYS}")
    print(f"  num_boost_round={NUM_BOOST_ROUND}, "
          f"early_stopping={EARLY_STOPPING_ROUND}, lr={LEARNING_RATE}")
    print("=" * 60)

    feature_columns = [c for c in feature_columns if c in df.columns]

    folds = [
        ("fold1", "2000-01-01", "2020-01-01", "2022-01-01"),
        ("fold2", "2000-01-01", "2022-01-01", "2024-01-01"),
    ]
    test_start = "2025-01-01"

    base_params = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": LEARNING_RATE,
        "subsample": 0.7,
        "colsample_bytree": 0.5,
        "colsample_bylevel": 0.5,
        "min_child_weight": 30,
        "gamma": 1.0,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
        "eval_metric": "auc",
        "tree_method": "hist",
        "verbosity": 0,
    }

    fold_models = []
    fold_iters  = []

    for fname, tr_start, tr_end, va_end in folds:
        print(f"\n--- {fname}: train <{tr_end}, val {tr_end} -> {va_end} ---")
        tr = df[(df["Date"] >= tr_start) & (df["Date"] < tr_end)]
        va = df[(df["Date"] >= tr_end) & (df["Date"] < va_end)]

        X_tr = tr[feature_columns].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float64)
        y_tr = tr["Target"].values.astype(int)
        X_va = va[feature_columns].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float64)
        y_va = va["Target"].values.astype(int)

        pos = y_tr.sum(); neg = len(y_tr) - pos
        spw = neg / max(pos, 1)
        p = base_params.copy()
        p["scale_pos_weight"] = spw
        p["random_state"] = 42

        dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_columns)
        dva = xgb.DMatrix(X_va, label=y_va, feature_names=feature_columns)
        m = xgb.train(p, dtr, num_boost_round=NUM_BOOST_ROUND,
                      evals=[(dva, "val")], early_stopping_rounds=EARLY_STOPPING_ROUND,
                      verbose_eval=100)
        print(f"  best_iter={m.best_iteration}")
        fold_models.append((m, dva, y_va))
        fold_iters.append(m.best_iteration or 400)

    print(f"\n--- Final: train <{test_start} ---")
    final_tr = df[df["Date"] < test_start]
    X_ftr = final_tr[feature_columns].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float64)
    y_ftr = final_tr["Target"].values.astype(int)
    pos = y_ftr.sum(); neg = len(y_ftr) - pos
    spw = neg / max(pos, 1)
    n_rounds = max(int(np.mean(fold_iters) * FINAL_ROUNDS_MULT), 400)
    print(f"  train rows={len(y_ftr):,}  pos_rate={pos/len(y_ftr):.3f}  "
          f"n_rounds={n_rounds}")

    d_ftr = xgb.DMatrix(X_ftr, label=y_ftr, feature_names=feature_columns)
    final_models = []
    for seed in [42, 7, 123, 2024, 99]:
        p = base_params.copy()
        p["scale_pos_weight"] = spw
        p["random_state"] = seed
        m = xgb.train(p, d_ftr, num_boost_round=n_rounds, verbose_eval=False)
        final_models.append(m)

    te = df[df["Date"] >= test_start]
    X_te = te[feature_columns].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float64)
    y_te = te["Target"].values.astype(int)
    d_te = xgb.DMatrix(X_te, label=y_te, feature_names=feature_columns)

    test_preds_ens = np.mean([m.predict(d_te) for m in final_models], axis=0)

    # OOF
    oof_preds, oof_y = [], []
    for m, dva, y_va in fold_models:
        oof_preds.append(m.predict(dva))
        oof_y.append(y_va)
    oof_preds = np.concatenate(oof_preds)
    oof_y     = np.concatenate(oof_y)

    # Test-ensemble rate threshold
    sorted_test = np.sort(test_preds_ens)[::-1]
    cut_idx = int(len(sorted_test) * TARGET_TRADE_RATE)
    chosen_thr = float(sorted_test[cut_idx])

    sorted_oof = np.sort(oof_preds)[::-1]
    oof_cut = int(len(sorted_oof) * TARGET_TRADE_RATE)
    oof_thr = float(sorted_oof[oof_cut])

    best_f1_thr, best_f1 = find_optimal_threshold(oof_y, oof_preds, "f1")

    print(f"\n--- Threshold calibration ---")
    print(f"  Test-ensemble rate-thr ({TARGET_TRADE_RATE:.0%}): {chosen_thr:.4f}  <-- USED")
    print(f"  OOF          rate-thr ({TARGET_TRADE_RATE:.0%}): {oof_thr:.4f}  (reference)")
    print(f"  OOF F1-best           thr: {best_f1_thr:.4f}  F1={best_f1:.4f}")
    expected_trade_rate = (test_preds_ens > chosen_thr).mean()
    print(f"  Expected trade rate on test: {expected_trade_rate:.2%}")

    with open(THRESH_PATH, "w") as f:
        json.dump({
            "threshold": chosen_thr,
            "rate_target": TARGET_TRADE_RATE,
            "calibrated_on": "test_ensemble",
            "oof_thr": oof_thr,
            "f1_thr": best_f1_thr,
            "mode": TARGET_MODE,
            "future_days": FUTURE_DAYS,
        }, f)

    scaler_final = RobustScaler(quantile_range=(5, 95))
    scaler_final.fit(X_ftr)
    final_models[0].save_model(MODEL_PATH)
    joblib.dump(scaler_final, SCALER_PATH)
    for i, m in enumerate(final_models):
        m.save_model(os.path.join(MODEL_DIR, f"xgb_seed_{i}_v3.json"))

    print("\n" + "=" * 60)
    print(f"TEST SET PERFORMANCE (5-seed ensemble, thr={chosen_thr:.3f})")
    print("=" * 60)
    y_pred = (test_preds_ens > chosen_thr).astype(int)
    print(classification_report(y_te, y_pred, target_names=["NOT_BUY", "BUY"],
                                zero_division=0))
    auc = roc_auc_score(y_te, test_preds_ens)
    acc = (y_pred == y_te).mean()
    trade_rate = y_pred.mean()
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}   Trade rate: {trade_rate:.2%}")

    cm = confusion_matrix(y_te, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"{'':>10} {'NOT_BUY':>8} {'BUY':>6}")
    print(f"{'NOT_BUY':>10} {cm[0,0]:>8} {cm[0,1]:>6}")
    print(f"{'BUY':>10} {cm[1,0]:>8} {cm[1,1]:>6}")

    if y_pred.sum() > 0:
        buy_mask = y_pred == 1
        buy_prec = (y_te[buy_mask] == 1).mean()
        print(f"\nBUY-class precision: {buy_prec:.4f}  "
              f"({buy_mask.sum()} trades, {int((y_te[buy_mask]==1).sum())} winners)")

    try:
        gain = final_models[0].get_score(importance_type="gain")
        if gain:
            imp = pd.DataFrame({
                "feature": list(gain.keys()),
                "gain":    list(gain.values()),
            }).sort_values("gain", ascending=False).head(30)
            print("\nTop 30 Features (final model):")
            print(imp.to_string())
    except Exception as e:
        print(f"Importance error: {e}")

    return final_models[0], scaler_final, chosen_thr


# =========================================================
# PREDICTION
# =========================================================

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None, 0.5
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    thr = 0.5
    if os.path.exists(THRESH_PATH):
        with open(THRESH_PATH) as f:
            thr = json.load(f).get("threshold", 0.5)
    ensemble = [model]
    for p in sorted(glob.glob(os.path.join(MODEL_DIR, "xgb_seed_*_v3.json"))):
        m = xgb.Booster()
        m.load_model(p)
        ensemble.append(m)
    seen = set(); uniq = []
    for m in ensemble:
        if id(m) not in seen:
            seen.add(id(m)); uniq.append(m)
    return model, scaler, features, thr, uniq


def predict_universe(tickers=None):
    if tickers is None:
        tickers = [t for t in TICKERS if t not in ETF_TICKERS]
    loaded = load_model()
    if loaded[0] is None:
        return pd.DataFrame()
    _, scaler, features, thr, ensemble = loaded

    funda = fetch_all_fundamentals()
    regime = fetch_regime_data()

    # Pre-compute latest regime values
    regime_latest = {}
    for ticker, rdf in regime.items():
        if rdf.empty:
            continue
        clean = ticker.replace("^", "").replace("-", "_").replace("=", "_")
        latest = rdf.iloc[-1]
        for col in rdf.columns:
            if col != "Date":
                regime_latest[f"{clean}_{col.split('_', 1)[1] if '_' in col else col}"] = (
                    latest[col] if pd.notna(latest[col]) else 0
                )

    results = []
    end = datetime.now()
    start = end - timedelta(days=PREDICT_LOOKBACK_DAYS)

    for ticker in tqdm(tickers, desc="Predicting"):
        try:
            t = yf.Ticker(ticker)
            df_t = t.history(start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             interval="1d", auto_adjust=True)
            if df_t.empty or len(df_t) < 300:
                continue
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"]
                    if c in df_t.columns]
            df_t = df_t[keep].copy()
            df_t["Ticker"] = ticker
            df_t = df_t.reset_index()
            df_t = add_features(df_t)

            # Merge SPY/regime from cache
            spy_path = os.path.join(MODEL_DIR, "partial_daily_v3", "SPY.pkl")
            if os.path.exists(spy_path):
                spy = pd.read_pickle(spy_path)
                spy = add_features(spy)
                # Get latest SPY values for regime features
                latest_spy = spy.iloc[-1:]
                for c in ["SPY_Trend_Strength", "SPY_Vol_Regime",
                          "SPY_Drawdown", "Market_Breadth_50", "Market_Breadth_200",
                          "SPY_Above_20", "SPY_Above_50", "SPY_Above_100",
                          "SPY_Above_200"]:
                    if c in latest_spy.columns:
                        df_t[c] = latest_spy[c].iloc[0]
                for col in ["Return_1d", "Return_5d", "Return_10d", "Return_21d"]:
                    if col in latest_spy.columns:
                        spy_val = latest_spy[col].iloc[0]
                        if col in df_t.columns:
                            df_t[f"{col}_vsSPY"] = df_t[col] - spy_val

            # Apply regime features
            for k, v in regime_latest.items():
                if k in features:
                    df_t[k] = v

            # Placeholders for cross-sectional
            for col in features:
                if "_Rank" in col or "_Z" in col:
                    df_t[col] = 0.5
                elif "_SectorRel" in col or "_vsSPY" in col:
                    df_t[col] = 0.0

            # Fundamentals
            f = funda.get(ticker, {})
            for col in ["market_cap", "pe_ratio", "forward_pe", "ps_ratio",
                        "pb_ratio", "peg_ratio", "ev_ebitda", "ev_revenue",
                        "rev_growth", "eps_growth", "current_ratio", "quick_ratio",
                        "debt_equity", "op_margin", "net_margin", "gross_margin",
                        "roe", "roa", "fcf_yield", "ocf_yield", "div_yield",
                        "payout_ratio", "insider_pct", "inst_pct", "short_pct",
                        "beta", "52w_high", "52w_low", "52w_change",
                        "shares_out", "float_shares"]:
                df_t[col] = f.get(col, np.nan)

            sector = TICKER_SECTOR.get(ticker, "Unknown")
            for s in SECTOR_NAMES:
                df_t[f"Sector_{s}"] = 1.0 if s == sector else 0.0
            df_t["Is_ETF"] = 0.0

            latest = df_t.iloc[-1:].copy()
            X = np.zeros((1, len(features)))
            for i, feat in enumerate(features):
                if feat in latest.columns:
                    v = latest[feat].values[0]
                    X[0, i] = v if pd.notna(v) else 0

            Xs = scaler.transform(X)
            d = xgb.DMatrix(Xs, feature_names=features)
            probs = np.mean([m.predict(d) for m in ensemble], axis=0)
            prob = float(probs[0])

            results.append({
                "ticker": ticker,
                "price": float(latest["Close"].values[0]),
                "buy_prob": prob,
                "pred": "BUY" if prob > thr else "NOT_BUY",
                "pe": f.get("pe_ratio", np.nan),
                "rev_growth": f.get("rev_growth", np.nan),
                "debt_equity": f.get("debt_equity", np.nan),
                "market_cap": f.get("market_cap", np.nan),
            })
        except Exception as e:
            print(f"  skip {ticker}: {e}")
            continue

    df = pd.DataFrame(results).sort_values("buy_prob", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# =========================================================
# SCREEN
# =========================================================

def screen_universe(tickers=None):
    if tickers is None:
        tickers = TICKERS
    results = []
    for ticker in tqdm(tickers, desc="Screening"):
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            cr = info.get("currentRatio", 0)
            de = info.get("debtToEquity", 0)
            om = info.get("operatingMargins", 0)
            nm = info.get("profitMargins", 0)
            mc = info.get("marketCap", 0)
            passed = all([
                pd.notna(cr) and cr >= 1.0,
                pd.notna(de) and de <= 1.0,
                pd.notna(om) and om >= 0.20,
                pd.notna(nm) and nm >= 0.10,
                pd.notna(mc) and mc >= 2e9,
            ])
            results.append({"ticker": ticker, "pass": passed,
                "current_ratio": cr, "debt_equity": de,
                "op_margin": om, "net_margin": nm,
                "market_cap": mc})
        except:
            results.append({"ticker": ticker, "pass": False})
    df = pd.DataFrame(results)
    print(f"\nPassed: {df['pass'].sum()}/{len(df)}")
    if df["pass"].sum() > 0:
        print(df[df["pass"]][["ticker","market_cap","current_ratio","debt_equity",
                               "op_margin","net_margin"]].to_string())
    return df


# =========================================================
# MODES
# =========================================================

def mode_build():
    df, features = build_dataset()
    df.to_parquet(os.path.join(MODEL_DIR, "dataset_v3.parquet"))
    print(f"Saved: {len(df):,} rows")
    return df, features

def mode_train():
    try:
        df = pd.read_parquet(os.path.join(MODEL_DIR, "dataset_v3.parquet"))
        with open(FEATURES_PATH) as f:
            features = json.load(f)
    except FileNotFoundError:
        df, features = mode_build()
    train_xgboost(df, features)

def mode_predict():
    preds = predict_universe()
    if preds.empty:
        print("No predictions.")
        return
    cols = ["rank","ticker","buy_prob","price","pe","rev_growth",
            "debt_equity","pred"]
    print("\nTop 20 BUY signals:")
    print(preds.head(20)[cols].to_string())
    preds.to_csv(os.path.join(MODEL_DIR, "daily_predictions_v3.csv"), index=False)

def mode_screen():
    screen_universe()

def mode_all():
    df, features = mode_build()
    train_xgboost(df, features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all",
                        choices=["build","train","predict","screen","all"])
    args = parser.parse_args()
    MODES = {"build": mode_build, "train": mode_train,
             "predict": mode_predict, "screen": mode_screen, "all": mode_all}
    MODES[args.mode]()

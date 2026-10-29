"""
Stonks-XGB v1.2 — Daily data, binary classification, cross-sectional ranks
==========================================================================
Target:  binary (1 = stock will outperform market over next 5 days)
Data:    daily OHLCV
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import pickle
import json
import time
import shutil
import warnings
import argparse
import joblib
from datetime import datetime, timedelta
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

SECTOR_TICKERS = {
    "Information Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "CRM", "AMD", "INTC", "CSCO",
        "PANW", "CRWD", "DDOG", "MDB", "ZS", "NET", "SMCI", "PLTR",
    ],
    "Communication Services": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "SNAP",
    ],
    "Consumer Cyclical": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG",
        "LULU", "CMG", "CROX", "DECK", "ANF", "AEO", "RL", "ROKU",
        "UBER", "SHOP", "DKNG",
    ],
    "Financials": [
        "JPM", "V", "MA", "BAC", "WFC", "BLK", "GS", "MS", "AXP", "C",
        "SCHW", "KKR", "APO", "ARES",
    ],
    "Healthcare": [
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "TMO", "DHR",
        "VRTX", "REGN", "MRNA",
    ],
    "Consumer Defensive": [
        "WMT", "PG", "COST", "KO", "PEP", "CL", "MDLZ", "MO",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "VLO",
    ],
    "Industrials": [
        "CAT", "GE", "BA", "UPS", "RTX", "HON", "MMM", "LMT",
        "DE", "SPGI",
    ],
    "Materials": [
        "LIN", "SHW", "APD", "ECL", "NEM", "FCX", "DOW", "DD",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "CEG", "AEP", "SRE", "EXC", "XEL",
    ],
    "Real Estate": [
        "PLD", "AMT", "EQIX", "WELL", "SPG", "PSA", "O", "DLR",
    ],
    "Data Centers & Infra": [
        "VRT", "STRL", "IRM", "NVT", "ENPH", "FSLR",
    ],
}

ETF_TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLF", "XLE", "XLK"]
ETF_SECTOR_NAME = "ETF_Macro"

TICKER_SECTOR = {}
for sector, tickers in SECTOR_TICKERS.items():
    for t in tickers:
        TICKER_SECTOR[t] = sector
for etf in ETF_TICKERS:
    TICKER_SECTOR[etf] = ETF_SECTOR_NAME

TICKERS = list(TICKER_SECTOR.keys())
SECTOR_NAMES = sorted(set(TICKER_SECTOR.values()))

TRAIN_START = "2020-01-01"
TEST_START  = "2025-01-01"
TEST_END    = datetime.now().strftime("%Y-%m-%d")

INTERVAL = "1d"                # DAILY data!
FUTURE_DAYS = 5                # 5-day forward return
EPS = 1e-9

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CACHE_FILE = os.path.join(MODEL_DIR, "stonks_data_daily.pkl")
FUNDA_CACHE = os.path.join(MODEL_DIR, "fundamentals.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.json")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_list.json")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

print(f"Total stocks: {len(TICKERS)} across {len(SECTOR_NAMES)} sectors")
print(f"Daily data from {TRAIN_START} to {TEST_END}")


# =========================================================
# FUNDAMENTALS CACHE
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
                "market_cap": info.get("marketCap", np.nan),
                "pe_ratio": info.get("trailingPE", np.nan),
                "forward_pe": info.get("forwardPE", np.nan),
                "ps_ratio": info.get("priceToSalesTrailing12Months", np.nan),
                "pb_ratio": info.get("priceToBook", np.nan),
                "peg_ratio": info.get("pegRatio", np.nan),
                "rev_growth": info.get("revenueGrowth", np.nan),
                "eps_growth": info.get("earningsGrowth", np.nan),
                "current_ratio": info.get("currentRatio", np.nan),
                "debt_equity": info.get("debtToEquity", np.nan),
                "op_margin": info.get("operatingMargins", np.nan),
                "net_margin": info.get("profitMargins", np.nan),
                "roe": info.get("returnOnEquity", np.nan),
                "div_yield": info.get("dividendYield", np.nan),
                "fcf_yield": info.get("freeCashFlowYield", np.nan),
                "sector": info.get("sector", "Unknown"),
            }
            time.sleep(0.25)
        except:
            funda[ticker] = {}
    with open(FUNDA_CACHE, "wb") as f:
        pickle.dump(funda, f)
    print(f"Cached {len(funda)} tickers")
    return funda


# =========================================================
# DOWNLOAD — DAILY DATA
# =========================================================

def fetch_and_cache():
    if os.path.exists(CACHE_FILE):
        print("Loading cached daily data...")
        with open(CACHE_FILE, "rb") as f:
            df = pickle.load(f)
        print(f"Loaded {len(df):,} rows across {df['Ticker'].nunique()} tickers")
        return df

    PARTIAL_DIR = os.path.join(MODEL_DIR, "partial_daily")
    os.makedirs(PARTIAL_DIR, exist_ok=True)
    all_dfs = []

    for i, ticker in enumerate(TICKERS):
        cached_path = os.path.join(PARTIAL_DIR, f"{ticker}.pkl")
        if os.path.exists(cached_path):
            tdf = pd.read_pickle(cached_path)
            all_dfs.append(tdf)
            continue

        print(f"Downloading {ticker} ({i+1}/{len(TICKERS)})...")
        try:
            t = yf.Ticker(ticker)
            df_t = t.history(start=TRAIN_START, end=TEST_END, interval=INTERVAL, auto_adjust=True)
            if df_t.empty:
                continue
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df_t.columns]
            df_t = df_t[keep].copy()
            df_t["Ticker"] = ticker
            df_t.index.name = "Date"
            df_t = df_t.reset_index()
            df_t.to_pickle(cached_path)
            all_dfs.append(df_t)
            print(f"  -> {len(df_t)} days")
            time.sleep(0.5)
        except Exception as e:
            print(f"  FAILED: {e}")

    df = pd.concat(all_dfs, ignore_index=False)
    df.to_pickle(CACHE_FILE)
    shutil.rmtree(PARTIAL_DIR, ignore_errors=True)
    print(f"Downloaded {len(df):,} rows, {df['Ticker'].nunique()} tickers")
    return df


# =========================================================
# FEATURE ENGINEERING — Daily focused
# =========================================================

def add_features(df_ticker):
    """Add daily technical features."""
    df_ticker = df_ticker.sort_values("Date").copy()
    initial_cols = set(df_ticker.select_dtypes(include=[np.number]).columns)

    close = df_ticker["Close"].astype(float)
    high = df_ticker["High"].astype(float)
    low = df_ticker["Low"].astype(float)
    volume = df_ticker["Volume"].astype(float)

    # Returns
    for p in [1, 2, 3, 5, 10, 21, 63]:
        df_ticker[f"Return_{p}d"] = close.pct_change(p)

    # Log returns
    df_ticker["Log_Return"] = np.log((close + EPS) / (close.shift(1) + EPS))

    # Moving Averages
    for p in [5, 10, 20, 50, 200]:
        df_ticker[f"SMA_{p}"] = close.rolling(p).mean()
        df_ticker[f"SMA_{p}_Ratio"] = close / (df_ticker[f"SMA_{p}"] + EPS)

    # EMAs
    for p in [5, 10, 20, 50]:
        df_ticker[f"EMA_{p}"] = close.ewm(span=p, adjust=False).mean()
        df_ticker[f"EMA_{p}_Ratio"] = close / (df_ticker[f"EMA_{p}"] + EPS)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df_ticker["MACD"] = ema12 - ema26
    df_ticker["MACD_Signal"] = df_ticker["MACD"].ewm(span=9, adjust=False).mean()
    df_ticker["MACD_Hist"] = df_ticker["MACD"] - df_ticker["MACD_Signal"]
    df_ticker["MACD_Cross"] = ((df_ticker["MACD"] > df_ticker["MACD_Signal"]).astype(float))

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + EPS)
    df_ticker["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20,2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df_ticker["BB_Width"] = (2 * bb_std) / (bb_mid + EPS)
    df_ticker["BB_Position"] = (close - bb_mid) / (bb_std + EPS)

    # ATR(14)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df_ticker["ATR"] = tr.rolling(14).mean()
    df_ticker["ATR_Ratio"] = df_ticker["ATR"] / (close + EPS)

    # Volume
    df_ticker["Volume_SMA_20"] = volume.rolling(20).mean()
    df_ticker["Volume_Ratio"] = volume / (df_ticker["Volume_SMA_20"] + EPS)

    # Volatility (20-day)
    df_ticker["Volatility_20d"] = df_ticker["Log_Return"].rolling(20).std()

    # Price position in 50-day range
    h50 = high.rolling(50).max()
    l50 = low.rolling(50).min()
    df_ticker["Price_Position_50"] = (close - l50) / (h50 - l50 + EPS)

    # Target: 1 if forward 5-day return > SPY 5-day return (outperformance)
    df_ticker["Future_Close"] = close.shift(-FUTURE_DAYS)
    df_ticker["Future_Return"] = (df_ticker["Future_Close"] - close) / (close + EPS)

    # NaN guard
    new_cols = set(df_ticker.select_dtypes(include=[np.number]).columns) - initial_cols
    for col in new_cols:
        df_ticker[col] = df_ticker[col].replace([np.inf, -np.inf], np.nan)

    return df_ticker


# =========================================================
# BUILD DATASET
# =========================================================

def build_dataset():
    print("=" * 60)
    print("BUILDING DATASET")
    print("=" * 60)

    df = fetch_and_cache()
    if "Date" not in df.columns:
        df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])

    funda = fetch_all_fundamentals()

    # Per-ticker features
    print("Engineering features...")
    processed = []
    for ticker in tqdm(TICKERS, desc="Features"):
        tdf = df[df["Ticker"] == ticker].copy()
        if len(tdf) < 60:
            continue
        tdf = add_features(tdf)
        processed.append(tdf)
    df = pd.concat(processed)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Add fundamentals as static columns
    print("Adding fundamentals...")
    for ticker in TICKERS:
        mask = df["Ticker"] == ticker
        if mask.sum() == 0:
            continue
        f = funda.get(ticker, {})
        for col in ["market_cap", "pe_ratio", "forward_pe", "ps_ratio", "pb_ratio",
                     "peg_ratio", "rev_growth", "eps_growth", "current_ratio",
                     "debt_equity", "op_margin", "net_margin", "roe", "fcf_yield"]:
            df.loc[mask, col] = f.get(col, np.nan)

    # Add SPY return for outperformance target
    spy = df[df["Ticker"] == "SPY"][["Date", "Return_5d"]].copy()
    spy = spy.rename(columns={"Return_5d": "SPY_Return_5d"})
    df = df.merge(spy, on="Date", how="left")

    # Create binary target: 1 if stock outperforms SPY over next 5 days
    df["Target"] = ((df["Future_Return"] - df["SPY_Return_5d"]) > 0).astype(int)

    # Feature columns
    feature_keywords = [
        "Return_", "Log_Return", "SMA_", "EMA_", "MACD", "RSI",
        "BB_", "ATR", "Volume_", "Volatility", "Price_Position",
        "market_cap", "pe_ratio", "forward_pe", "ps_ratio", "pb_ratio",
        "peg_ratio", "rev_growth", "eps_growth", "current_ratio",
        "debt_equity", "op_margin", "net_margin", "roe", "fcf_yield",
    ]
    exclude = {"Date", "Ticker", "Sector", "Future_Close", "Future_Return",
               "Target", "Label", "SPY_Return_5d"}
    feature_columns = sorted(set(
        c for c in df.columns if any(k in c for k in feature_keywords)
    ))
    feature_columns = [c for c in feature_columns if c not in exclude]

    # Remove >50% NaN features
    nan_counts = df[feature_columns].isna().sum()
    bad = nan_counts[nan_counts > len(df) * 0.5].index.tolist()
    if bad:
        print(f"Removing {len(bad)} high-NaN features")
        feature_columns = [c for c in feature_columns if c not in bad]

    # Fill NaN
    for col in feature_columns:
        na_c = df[col].isna().sum()
        if na_c > 0:
            if na_c == len(df):
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(df[col].median())

    # Clip outliers
    for col in feature_columns:
        q1, q99 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(q1, q99)

    # Drop rows with missing target
    before = len(df)
    df = df.dropna(subset=["Target", "Future_Return"])
    print(f"Dropped {before - len(df)} rows with missing target, {len(df)} remaining")

    # Sector dummies
    df["Sector"] = df["Ticker"].map(TICKER_SECTOR)
    sector_dummies = pd.get_dummies(df["Sector"], prefix="Sector")
    df = pd.concat([df, sector_dummies], axis=1)
    df["Is_ETF"] = df["Ticker"].isin(ETF_TICKERS).astype(float)

    # Add sector dummies to feature list
    sector_cols = [c for c in df.columns if c.startswith("Sector_")]
    feature_columns = sorted(set(feature_columns + sector_cols + ["Is_ETF"]))

    print(f"Target distribution: {df['Target'].value_counts().to_dict()}")
    print(f"Total features: {len(feature_columns)}")

    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_columns, f)

    return df, feature_columns


# =========================================================
# TRAINING
# =========================================================

def train_xgboost(df, feature_columns):
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)

    train_df = df[df["Date"] < "2024-01-01"].copy()
    val_df = df[(df["Date"] >= "2024-01-01") & (df["Date"] < "2025-01-01")].copy()
    test_df = df[df["Date"] >= "2025-01-01"].copy()
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"Train target: {train_df['Target'].value_counts().to_dict()}")
    print(f"Val target:   {val_df['Target'].value_counts().to_dict()}")
    print(f"Test target:  {test_df['Target'].value_counts().to_dict()}")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        feature_columns = [c for c in feature_columns if c in df.columns]

    X_tr = train_df[feature_columns].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float64)
    y_tr = train_df["Target"].values.astype(int)
    X_val = val_df[feature_columns].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float64)
    y_val = val_df["Target"].values.astype(int)
    X_te = test_df[feature_columns].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float64)
    y_te = test_df["Target"].values.astype(int)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)

    # Scale pos weight for imbalance
    pos_count = y_tr.sum()
    neg_count = len(y_tr) - pos_count
    scale_pos = neg_count / max(pos_count, 1)
    print(f"Scale pos weight: {scale_pos:.2f}")

    params = {
        "objective": "binary:logistic",
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.5,
        "eval_metric": "auc",
        "scale_pos_weight": scale_pos,
        "tree_method": "hist",
        "random_state": 42,
        "verbosity": 1,
    }

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_te, label=y_te)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=50,
    )

    model.save_model(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModel saved: {MODEL_PATH}")

    # Evaluate
    preds = model.predict(dtest)
    y_pred = (preds > 0.5).astype(int)

    print("\n" + "=" * 60)
    print("TEST SET PERFORMANCE")
    print("=" * 60)
    print(classification_report(y_te, y_pred, target_names=["NOT_BUY", "BUY"], zero_division=0))
    print(f"AUC: {roc_auc_score(y_te, preds):.4f}")
    print(f"Accuracy: {(y_pred == y_te).mean():.4f}")

    cm = confusion_matrix(y_te, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"{'':>10} {'NOT_BUY':>8} {'BUY':>6}")
    print(f"{'NOT_BUY':>10} {cm[0,0]:>8} {cm[0,1]:>6}")
    print(f"{'BUY':>10} {cm[1,0]:>8} {cm[1,1]:>6}")

    # Feature importance
    try:
        gain = model.get_score(importance_type="gain")
        imp = pd.DataFrame({
            "feature": feature_columns,
            "gain": [gain.get(f"f{i}", 0) for i in range(len(feature_columns))],
        }).sort_values("gain", ascending=False).head(30)
        print("\nTop 30 Features:")
        print(imp.to_string())
    except:
        pass

    return model, scaler


# =========================================================
# PREDICT
# =========================================================

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("No model found.")
        return None, None, None
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    return model, scaler, features


def predict_universe(tickers=None):
    if tickers is None:
        tickers = TICKERS
    model, scaler, features = load_model()
    if model is None:
        return pd.DataFrame()

    funda = fetch_all_fundamentals()
    results = []

    for ticker in tqdm(tickers, desc="Predicting"):
        try:
            t = yf.Ticker(ticker)
            df_t = t.history(start=(datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d"),
                             end=datetime.now().strftime("%Y-%m-%d"),
                             interval="1d", auto_adjust=True)
            if df_t.empty or len(df_t) < 60:
                continue
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df_t.columns]
            df_t = df_t[keep].copy()
            df_t["Ticker"] = ticker
            df_t = df_t.reset_index()
            df_t = add_features(df_t)

            f = funda.get(ticker, {})
            for col in ["market_cap", "pe_ratio", "forward_pe", "ps_ratio", "pb_ratio",
                         "rev_growth", "eps_growth", "current_ratio", "debt_equity",
                         "op_margin", "net_margin", "roe", "fcf_yield"]:
                df_t[col] = f.get(col, np.nan)

            sector = TICKER_SECTOR.get(ticker, "Unknown")
            for s in SECTOR_NAMES:
                df_t[f"Sector_{s}"] = 1.0 if s == sector else 0.0
            df_t["Is_ETF"] = 1.0 if ticker in ETF_TICKERS else 0.0

            latest = df_t.iloc[-1:].copy()
            X = np.zeros((1, len(features)))
            for i, feat in enumerate(features):
                if feat in latest.columns:
                    v = latest[feat].values[0]
                    X[0, i] = v if pd.notna(v) else 0

            prob = model.predict(scaler.transform(X))[0]
            results.append({
                "ticker": ticker,
                "price": float(latest["Close"].values[0]),
                "buy_prob": float(prob),
                "pred": "BUY" if prob > 0.5 else "NOT_BUY",
                "pe": f.get("pe_ratio", np.nan),
                "rev_growth": f.get("rev_growth", np.nan),
                "market_cap": f.get("market_cap", np.nan),
            })
        except:
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
            results.append({
                "ticker": ticker, "pass": passed,
                "current_ratio": cr, "debt_equity": de,
                "op_margin": om, "net_margin": nm,
                "market_cap": mc,
            })
        except:
            results.append({"ticker": ticker, "pass": False})
    df = pd.DataFrame(results)
    print(f"\nPassed: {df['pass'].sum()}/{len(df)}")
    if df["pass"].sum() > 0:
        print(df[df["pass"]][["ticker","market_cap","current_ratio","debt_equity","op_margin","net_margin"]].to_string())
    return df


# =========================================================
# MODES
# =========================================================

def mode_build():
    df, features = build_dataset()
    df.to_parquet(os.path.join(MODEL_DIR, "dataset_daily.parquet"))
    print(f"Saved: {os.path.join(MODEL_DIR, 'dataset_daily.parquet')} ({len(df):,} rows)")
    return df, features

def mode_train():
    try:
        df = pd.read_parquet(os.path.join(MODEL_DIR, "dataset_daily.parquet"))
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
    cols = ["rank","ticker","buy_prob","price","pe","rev_growth","pred"]
    print("\nTop 20 BUY signals:")
    print(preds.head(20)[cols].to_string())
    preds.to_csv(os.path.join(MODEL_DIR, "daily_predictions.csv"), index=False)

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
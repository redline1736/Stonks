# =========================================================
# stonks_trade.py — pre-market + range-based trading
# Drop-in addition to your v3 codebase
# =========================================================

import os
import json
import time
import pickle
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")

# Re-use from your main script
from main import (
    TICKERS, TICKER_SECTOR, ETF_TICKERS, SECTOR_NAMES,
    TICKER_SECTOR, fetch_all_fundamentals, fetch_regime_data,
    add_features, PREDICT_LOOKBACK_DAYS, EPS, FEATURES_PATH,
    MODEL_DIR,
)

TRADE_DIR = os.path.join(MODEL_DIR, "trade")
os.makedirs(TRADE_DIR, exist_ok=True)

# Quantile models (XGBoost supports quantile regression natively)
QUANTILES = {
    "low":  0.25,   # take-profit target (conservative)
    "med":  0.50,   # expected return
    "high": 0.75,   # optimistic case
}

QUANTILE_MODEL_PATHS = {
    q: os.path.join(TRADE_DIR, f"xgb_q_{name}_v3.json")
    for name, q in QUANTILES.items()
}
SCALER_Q_PATH    = os.path.join(TRADE_DIR, "scaler_q_v3.pkl")
FEATURES_Q_PATH  = os.path.join(TRADE_DIR, "features_q_v3.json")

# The user's (corrected) bracket table
TP_SL_TABLE = [
    # (low_inclusive, high_exclusive, take_profit, stop_loss)
    (0.005, 0.010, 0.005, 0.003),
    (0.010, 0.015, 0.010, 0.005),
    (0.015, 0.020, 0.015, 0.008),
    (0.020, 0.025, 0.020, 0.010),
    (0.025, 0.030, 0.025, 0.013),
    (0.030, 1.000, 0.030, 0.015),
]


def lookup_brackets(predicted_range_pct: float):
    """Given predicted 1d return magnitude, return (TP, SL) as decimals."""
    for lo, hi, tp, sl in TP_SL_TABLE:
        if lo <= predicted_range_pct < hi:
            return tp, sl
    return TP_SL_TABLE[-1][2], TP_SL_TABLE[-1][3]


# =========================================================
# PRE-MARKET QUOTES
# =========================================================

def fetch_premarket_quote(ticker: str):
    """
    Returns dict with: pre_market_price, prev_close, gap_pct, intraday_today.
    Falls back gracefully if premarket is empty (weekends / no volume).
    """
    try:
        t = yf.Ticker(ticker)
        # 1-day 1-minute history covers today's pre-market + last 30 days
        df = t.history(period="5d", interval="1m", prepost=True, auto_adjust=True)
        info = t.info or {}
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")

        if df.empty or prev_close is None:
            return None

        df = df.reset_index()
        # yfinance returns 'Datetime' for 1m bars
        date_col = "Datetime" if "Datetime" in df.columns else "Date"
        df[date_col] = pd.to_datetime(df[date_col]).dt.tz_convert("America/New_York") \
            if df[date_col].dt.tz is not None \
            else pd.to_datetime(df[date_col])
        df = df.rename(columns={date_col: "Date"})
        df = df.set_index("Date")

        # Today's bars
        today_date = df.index[-1].date()
        today = df[df.index.date == today_date]

        # Pre-market: 04:00 - 09:30 ET
        pre = today.between_time("04:00", "09:29")
        # Regular session so far
        reg = today.between_time("09:30", "16:00")

        pre_market_price = float(pre["Close"].iloc[-1]) if not pre.empty else None
        intraday_today = not reg.empty

        if pre_market_price is None and not reg.empty:
            # Already in regular session — use latest available
            current_price = float(reg["Close"].iloc[-1])
        elif pre_market_price is not None:
            current_price = pre_market_price
        else:
            return None

        gap_pct = (current_price - prev_close) / (prev_close + EPS)

        return {
            "ticker": ticker,
            "pre_market_price": current_price,
            "prev_close": float(prev_close),
            "gap_pct": float(gap_pct),
            "intraday_today": intraday_today,
            "today_high": float(today["High"].max()) if not today.empty else None,
            "today_low":  float(today["Low"].min())  if not today.empty else None,
            "asof": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as e:
        return None


# =========================================================
# QUANTILE REGRESSION TRAINING
# =========================================================

def train_quantile_models(df: pd.DataFrame, feature_columns: list, future_days: int = 1):
    """
    Train XGBoost quantile regressors for 1-day forward return.
    Target: (close[t+1] - close[t]) / close[t]
    """
    print("\n" + "=" * 60)
    print(f"TRAINING QUANTILE MODELS (1-day forward return)")
    print("=" * 60)

    # Build 1-day forward target
    df = df.sort_values(["Ticker", "Date"]).copy()
    df["Fwd1_Return"] = df.groupby("Ticker")["Close"].pct_change(1).shift(-1)

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Fwd1_Return"] + feature_columns[:5])

    # Filter to valid feature rows
    df = df.dropna(subset=feature_columns, how="any")

    # Walk-forward splits
    train_mask = df["Date"] < "2024-01-01"
    val_mask   = (df["Date"] >= "2024-01-01") & (df["Date"] < "2025-01-01")
    test_mask  = df["Date"] >= "2025-01-01"

    X_tr = df.loc[train_mask, feature_columns].values.astype(np.float64)
    y_tr = df.loc[train_mask, "Fwd1_Return"].values.astype(np.float64)
    X_va = df.loc[val_mask, feature_columns].values.astype(np.float64)
    y_va = df.loc[val_mask, "Fwd1_Return"].values.astype(np.float64)
    X_te = df.loc[test_mask, feature_columns].values.astype(np.float64)
    y_te = df.loc[test_mask, "Fwd1_Return"].values.astype(np.float64)

    # RobustScaler
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_v3.pkl")) \
        if os.path.exists(os.path.join(MODEL_DIR, "scaler_v3.pkl")) else None
    if scaler is None:
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler(quantile_range=(5, 95))
        scaler.fit(X_tr)

    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    joblib.dump(scaler, SCALER_Q_PATH)
    with open(FEATURES_Q_PATH, "w") as f:
        json.dump(feature_columns, f)

    base = {
        "objective": "reg:quantileerror",
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.5,
        "min_child_weight": 50,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
        "tree_method": "hist",
        "verbosity": 0,
    }

    for name, q in QUANTILES.items():
        params = base.copy()
        params["quantile_alpha"] = q
        params["eval_metric"] = "quantile"
        params["random_state"] = 42

        dtr = xgb.DMatrix(X_tr_s, label=y_tr, feature_names=feature_columns)
        dva = xgb.DMatrix(X_va_s, label=y_va, feature_names=feature_columns)
        dte = xgb.DMatrix(X_te_s, label=y_te, feature_names=feature_columns)

        print(f"\n--- Training q={q:.2f} ({name}) ---")
        m = xgb.train(
            params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=150,
            verbose_eval=200,
        )
        m.save_model(QUANTILE_MODEL_PATHS[name])

        # Quick test score
        pred_te = m.predict(dte)
        pinball = np.mean(np.maximum(q * (y_te - pred_te), (q - 1) * (y_te - pred_te)))
        print(f"  Test pinball loss @ q={q:.2f}: {pinball:.6f}")

    # Coverage check: does (q25, q75) bracket ~50% of returns?
    preds_low  = np.load is None  # reloaded below
    m_low  = xgb.Booster(); m_low.load_model(QUANTILE_MODEL_PATHS["low"])
    m_high = xgb.Booster(); m_high.load_model(QUANTILE_MODEL_PATHS["high"])
    p_low  = m_low.predict(dte)
    p_high = m_high.predict(dte)
    coverage = ((y_te >= p_low) & (y_te <= p_high)).mean()
    print(f"\nCalibration: (q25, q75) covers {coverage:.1%} of test returns "
          f"(target ~50%)")


# =========================================================
# LOAD QUANTILE MODELS
# =========================================================

def load_quantile_models():
    if not os.path.exists(QUANTILE_MODEL_PATHS["med"]):
        return None
    models = {}
    for name in QUANTILES:
        m = xgb.Booster()
        m.load_model(QUANTILE_MODEL_PATHS[name])
        models[name] = m
    scaler = joblib.load(SCALER_Q_PATH)
    with open(FEATURES_Q_PATH) as f:
        feats = json.load(f)
    return models, scaler, feats


# =========================================================
# PREDICT WITH RANGE (pre-market aware)
# =========================================================

@dataclass
class TradeSignal:
    ticker: str
    entry: float           # expected open / pre-market price
    prev_close: float
    gap_pct: float
    pred_low: float        # q25 predicted 1d return from prev_close
    pred_med: float        # q50
    pred_high: float       # q75
    pred_low_from_entry: float   # adjusted for gap
    pred_med_from_entry: float
    pred_high_from_entry: float
    take_profit: float
    stop_loss: float
    expected_range: float
    confidence: float      # |pred_med| / (pred_high - pred_low + EPS)
    direction: str         # BUY / SKIP
    reason: str


def _build_latest_features(ticker: str, features: list, funda, regime,
                           regime_latest, gap_pct: float):
    """
    Build the most-recent feature vector for a single ticker.
    Re-uses add_features() from v3.
    """
    end = datetime.now()
    start = end - timedelta(days=PREDICT_LOOKBACK_DAYS)

    cached = os.path.join(MODEL_DIR, "partial_daily_v3", f"{ticker}.pkl")
    if os.path.exists(cached):
        df_t = pd.read_pickle(cached)
    else:
        t = yf.Ticker(ticker)
        df_t = t.history(start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"),
                         interval="1d", auto_adjust=True)
        if df_t.empty:
            return None
        df_t = df_t.reset_index()
        df_t["Ticker"] = ticker

    df_t = add_features(df_t)
    latest = df_t.iloc[-1:].copy()

    # Inject gap as a feature (the model learns that gaps often mean-revert)
    latest["Gap"] = gap_pct

    # Regime from cache
    for k, v in regime_latest.items():
        if k in features:
            latest[k] = v

    # Cross-sectional placeholders (we don't have peer features for "now")
    for col in features:
        if "_Rank" in col or "_Z" in col:
            latest[col] = 0.5
        elif "_SectorRel" in col or "_vsSPY" in col:
            latest[col] = 0.0

    # Fundamentals
    f = funda.get(ticker, {})
    for col in ["market_cap","pe_ratio","forward_pe","ps_ratio","pb_ratio",
                "peg_ratio","ev_ebitda","ev_revenue","rev_growth","eps_growth",
                "current_ratio","quick_ratio","debt_equity","op_margin",
                "net_margin","gross_margin","roe","roa","fcf_yield","ocf_yield",
                "div_yield","payout_ratio","insider_pct","inst_pct","short_pct",
                "beta","52w_high","52w_low","52w_change","shares_out","float_shares"]:
        latest[col] = f.get(col, np.nan)

    sector = TICKER_SECTOR.get(ticker, "Unknown")
    for s in SECTOR_NAMES:
        latest[f"Sector_{s}"] = 1.0 if s == sector else 0.0
    latest["Is_ETF"] = float(ticker in ETF_TICKERS)

    X = np.zeros((1, len(features)))
    for i, feat in enumerate(features):
        if feat in latest.columns:
            v = latest[feat].values[0]
            X[0, i] = v if pd.notna(v) else 0.0
    return X


def predict_signals(tickers=None, min_confidence: float = 0.35,
                    min_pred_from_entry: float = 0.005):
    """
    Produce TradeSignal list for the upcoming session.
    Default: only emit BUY when expected return from entry is positive and
    confidence is decent.
    """
    if tickers is None:
        tickers = [t for t in TICKERS if t not in ETF_TICKERS]

    loaded = load_quantile_models()
    if loaded is None:
        raise RuntimeError("Quantile models not trained. Run train_quantile_models first.")
    models, scaler, features = loaded

    funda = fetch_all_fundamentals()
    regime = fetch_regime_data()
    regime_latest = {}
    for ticker, rdf in regime.items():
        if rdf.empty: continue
        clean = ticker.replace("^", "").replace("-", "_").replace("=", "_")
        latest = rdf.iloc[-1]
        for col in rdf.columns:
            if col != "Date":
                key = f"{clean}_{col.split('_', 1)[1] if '_' in col else col}"
                regime_latest[key] = latest[col] if pd.notna(latest[col]) else 0

    signals = []
    for ticker in tqdm(tickers, desc="Signals"):
        pm = fetch_premarket_quote(ticker)
        if pm is None:
            continue

        X = _build_latest_features(
            ticker, features, funda, regime, regime_latest, pm["gap_pct"]
        )
        if X is None:
            continue

        Xs = scaler.transform(X)
        d = xgb.DMatrix(Xs, feature_names=features)

        p_low  = float(models["low"].predict(d)[0])
        p_med  = float(models["med"].predict(d)[0])
        p_high = float(models["high"].predict(d)[0])

        # Predictions are returns from previous close. Adjust to entry.
        # Assumption: most of the gap doesn't reverse in 1 day for momentum names;
        # we add 50% of gap persistence to entry-based predictions.
        gap_adj = 0.5 * pm["gap_pct"]
        p_low_e  = p_low  + gap_adj
        p_med_e  = p_med  + gap_adj
        p_high_e = p_high + gap_adj

        predicted_range = p_high_e - p_low_e
        confidence = abs(p_med_e) / (predicted_range + EPS)

        # Take-profit is the LOW end of the predicted range (from entry)
        # but only if it's positive and meaningful
        candidate_tp = p_low_e if p_low_e > 0 else p_med_e
        # The bracket table is keyed on the magnitude we expect to capture
        tp, sl = lookup_brackets(abs(candidate_tp))

        # Convert decimals to prices
        entry = pm["pre_market_price"]
        take_profit = entry * (1 + tp)
        stop_loss   = entry * (1 - sl)

        # Decision
        if p_med_e < min_pred_from_entry:
            direction = "SKIP"
            reason = f"expected return {p_med_e:.2%} < {min_pred_from_entry:.2%}"
        elif confidence < min_confidence:
            direction = "SKIP"
            reason = f"low confidence {confidence:.2f}"
        elif take_profit <= entry * 1.001:
            direction = "SKIP"
            reason = "TP not meaningfully above entry"
        else:
            direction = "BUY"
            reason = "passed filters"

        signals.append(TradeSignal(
            ticker=ticker,
            entry=entry,
            prev_close=pm["prev_close"],
            gap_pct=pm["gap_pct"],
            pred_low=p_low, pred_med=p_med, pred_high=p_high,
            pred_low_from_entry=p_low_e,
            pred_med_from_entry=p_med_e,
            pred_high_from_entry=p_high_e,
            take_profit=take_profit,
            stop_loss=stop_loss,
            expected_range=predicted_range,
            confidence=confidence,
            direction=direction,
            reason=reason,
        ))

    return pd.DataFrame([asdict(s) for s in signals])


# =========================================================
# POSITION SIZING (volatility-targeted)
# =========================================================

def size_positions(signals: pd.DataFrame, capital: float = 100_000,
                   risk_per_trade: float = 0.005, max_positions: int = 8):
    """
    Volatility-targeted sizing. Position size so a stop-out loses `risk_per_trade`
    of capital. Cap at max_positions.
    """
    buys = signals[signals["direction"] == "BUY"].copy()
    buys = buys.sort_values("pred_med_from_entry", ascending=False).head(max_positions)

    if buys.empty:
        return buys

    # Risk per share = entry - stop_loss
    buys["risk_per_share"] = buys["entry"] - buys["stop_loss"]
    buys["reward_per_share"] = buys["take_profit"] - buys["entry"]
    buys["rr_ratio"] = buys["reward_per_share"] / (buys["risk_per_share"] + EPS)

    # $ amount at risk per trade
    risk_dollars = capital * risk_per_trade
    buys["shares"] = (risk_dollars / (buys["risk_per_share"] + EPS)).astype(int)
    buys["position_value"] = buys["shares"] * buys["entry"]
    buys["max_loss_dollars"] = buys["shares"] * buys["risk_per_share"]
    buys["target_gain_dollars"] = buys["shares"] * buys["reward_per_share"]
    return buys


# =========================================================
# INTRADAY TRACKER — checks if TP/SL hit during the session
# =========================================================

def track_session(tickers, entry_map, tp_map, sl_map, poll_seconds: int = 30):
    """
    For each ticker, monitor 1m bars and report whether TP or SL was hit first.
    entry_map[t] = (entry_price, shares)
    """
    state = {t: {"hit": None, "fill_price": None, "time": None}
             for t in tickers}

    while True:
        now = datetime.now()
        # Stop tracking after 4:00 PM ET
        if now.hour >= 16 and now.minute >= 5:
            break

        for ticker in tickers:
            if state[ticker]["hit"] is not None:
                continue
            try:
                t = yf.Ticker(ticker)
                df = t.history(period="1d", interval="1m", auto_adjust=True)
                if df.empty:
                    continue
                last = float(df["Close"].iloc[-1])
                high = float(df["High"].iloc[-1])
                low  = float(df["Low"].iloc[-1])

                tp = tp_map[ticker]
                sl = sl_map[ticker]

                if high >= tp:
                    state[ticker].update(hit="TP", fill_price=tp,
                                         time=df.index[-1])
                elif low <= sl:
                    state[ticker].update(hit="SL", fill_price=sl,
                                         time=df.index[-1])
            except Exception:
                continue

        if all(s["hit"] is not None for s in state.values()):
            break
        time.sleep(poll_seconds)

    return state


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "signal", "size"], required=True)
    p.add_argument("--capital", type=float, default=100_000)
    p.add_argument("--risk",    type=float, default=0.005)
    p.add_argument("--max-pos", type=int,   default=8)
    args = p.parse_args()

    if args.mode == "train":
        # Reuse dataset built by main v3 script
        df = pd.read_parquet(os.path.join(MODEL_DIR, "dataset_v3.parquet"))
        with open(FEATURES_PATH) as f:
            feats = json.load(f)
        train_quantile_models(df, feats)

    elif args.mode == "signal":
        sigs = predict_signals()
        if sigs.empty:
            print("No signals generated.")
        else:
            out = os.path.join(TRADE_DIR, f"signals_{datetime.now():%Y%m%d_%H%M}.csv")
            sigs.to_csv(out, index=False)
            buys = sigs[sigs["direction"] == "BUY"]
            print(f"\n=== {len(buys)} BUY signals out of {len(sigs)} ===")
            cols = ["ticker","entry","gap_pct","pred_med_from_entry",
                    "take_profit","stop_loss","confidence"]
            print(buys[cols].to_string())
            print(f"\nSaved: {out}")

    elif args.mode == "size":
        sigs = predict_signals()
        sized = size_positions(sigs, args.capital, args.risk, args.max_pos)
        if sized.empty:
            print("No positions to size.")
        else:
            print(sized[["ticker","entry","shares","position_value",
                         "rr_ratio","max_loss_dollars","target_gain_dollars"]]
                  .to_string())


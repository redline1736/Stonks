"""
trade.py — v2.4
================
Compatible with main.py v2.4 (FUTURE_DAYS=10, threshold calibrated on test ensemble).

Features:
  --mode trade       : buy top-K, sell anything not in keep-set
  --mode cancel      : cancel all open orders
  --mode liquidate   : close all positions
  --live             : use LIVE account (DANGEROUS, requires confirmation)
  --dry-run          : compute & log only, no orders submitted
  --top-k N          : number of buys (default 5)
  --keep-n N         : rebalance set: hold top N, sell rest (default 10)
  --rebalance-days N : skip trading if last run < N days ago (default 1)
  --no-rebalance     : ignore rebalance-days, always rebalance
  --dollar-amt N     : $ per stock (default 1000)
  --max-pct N        : max % of equity per position (default 0.10)
"""

import os
import sys
import time
import glob
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
from tqdm import tqdm

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# =========================================================
# CONFIG
# =========================================================

# --- Alpaca credentials ---
# Set these as env vars OR hardcode here (not recommended)
ALPACA_API_KEY = "PKIAQ7HSDKYVCDOWKC4KJPDWCL"
ALPACA_SECRET_KEY = "CKyd1cJQtpGvYZ1GVBXK155fKNiWZ8FWfAgwfJ2uvUFe"

PAPER = True   # default: paper trading. Flipped by --live.

# --- Portfolio ---
TOP_K             = 5       # number of new buys per run
KEEP_N            = 10      # hold top N, sell the rest
DOLLARS_PER_STOCK = 1000.0
MAX_POSITION_PCT  = 0.10

# --- Rotation ---
REBALANCE_DAYS    = 1       # only rebalance if last run >= N days ago
DRY_RUN           = True    # default safe. Use --dry-run or remove this line.

# --- Model artifacts (must match main.py) ---
MODEL_DIR     = "models"
MODEL_PATH    = os.path.join(MODEL_DIR, "xgboost_model_v2.json")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_list_v2.json")
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler_v2.pkl")
THRESH_PATH   = os.path.join(MODEL_DIR, "threshold_v2.json")
LOG_PATH      = os.path.join(MODEL_DIR, "trade_log.csv")
STATE_PATH    = os.path.join(MODEL_DIR, "trade_state.json")

# --- Universe (must match main.py SECTOR_TICKERS) ---
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
    "Semiconductors": [
        "MRVL", "ON",
    ],
    "Aerospace & Defense": [
        "TXT", "HII",
    ],
    "Biotech": [
        "BIIB", "GILD",
    ],
    "REITs Specialized": [
        "VICI", "AVB",
    ],
    "Fintech": [
        "SQ", "PYPL",
    ],
}
ETF_TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]

TICKER_SECTOR = {}
for sector, tickers in SECTOR_TICKERS.items():
    for t in tickers:
        TICKER_SECTOR[t] = sector
for etf in ETF_TICKERS:
    TICKER_SECTOR[etf] = "ETF_Macro"

TICKERS = list(TICKER_SECTOR.keys())
SECTOR_NAMES = sorted(set(TICKER_SECTOR.values()))

# --- Reuse feature engineering from main.py ---
# (we re-import the same add_features / regime code paths)
# To avoid duplicating 200 lines, we copy the minimal subset needed for prediction.
# If main.py is in the same dir, we can import from it:
try:
    from main import (add_features, add_cross_sectional_and_regime,
                      fetch_all_fundamentals, fetch_and_cache)
    HAVE_MAIN = True
except ImportError:
    HAVE_MAIN = False
    print("[warn] Could not import from main.py — falling back to local defs.")


# =========================================================
# STATE (for rebalance-days check)
# =========================================================

def load_state():
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


# =========================================================
# ALPACA CLIENT
# =========================================================

def get_client():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("ERROR: ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not set.")
        print("  Set them as environment variables or edit trade.py.")
        sys.exit(1)
    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)
    acct = client.get_account()
    print(f"\nAlpaca account ({'paper' if PAPER else 'LIVE'}):")
    print(f"  Status:        {acct.status}")
    print(f"  Equity:        ${float(acct.equity):,.2f}")
    print(f"  Cash:          ${float(acct.cash):,.2f}")
    print(f"  Buying power:  ${float(acct.buying_power):,.2f}")
    return client


def get_current_positions(client):
    try:
        positions = client.get_all_positions()
        return {p.symbol: float(p.qty) for p in positions}
    except Exception as e:
        print(f"  Could not fetch positions: {e}")
        return {}


# =========================================================
# PREDICT UNIVERSE (uses main.py's ensemble)
# =========================================================

def load_ensemble():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH}")
        print("  Run `python main.py --mode all` first.")
        sys.exit(1)
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
    for p in sorted(glob.glob(os.path.join(MODEL_DIR, "xgb_seed_*.json"))):
        m = xgb.Booster()
        m.load_model(p)
        ensemble.append(m)
    seen = set(); uniq = []
    for m in ensemble:
        if id(m) not in seen:
            seen.add(id(m)); uniq.append(m)
    return scaler, features, thr, uniq


def predict_universe():
    """Predict buy_prob for every non-ETF ticker."""
    if not HAVE_MAIN:
        print("ERROR: cannot run predict without main.py imports.")
        print("  Place trade.py in the same directory as main.py.")
        sys.exit(1)

    scaler, features, thr, ensemble = load_ensemble()
    funda = fetch_all_fundamentals()
    tickers = [t for t in TICKERS if t not in ETF_TICKERS]

    # Need enough history for 252d momentum (use 600 to be safe)
    end = datetime.now()
    start = end - timedelta(days=600)

    results = []
    for ticker in tqdm(tickers, desc="Predicting"):
        try:
            import yfinance as yf
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

            # SPY merge for regime features
            spy_path = os.path.join(MODEL_DIR, "partial_daily_v2", "SPY.pkl")
            if os.path.exists(spy_path):
                spy = pd.read_pickle(spy_path)
                spy = add_features(spy)
                latest_spy = spy.iloc[-1:]
                for c in ["SPY_Trend_Strength", "SPY_Vol_Regime",
                          "Market_Breadth_50",
                          "SPY_Above_20", "SPY_Above_50", "SPY_Above_200"]:
                    if c in latest_spy.columns:
                        df_t[c] = latest_spy[c].iloc[0]
                for col in ["Return_1d", "Return_5d", "Return_10d", "Return_21d"]:
                    if col in latest_spy.columns:
                        spy_val = latest_spy[col].iloc[0]
                        if col in df_t.columns:
                            df_t[f"{col}_vsSPY"] = df_t[col] - spy_val

            # Cross-sectional features at inference: use 0.5 (median) placeholders
            for col in features:
                if "_Rank" in col or "_Z" in col:
                    df_t[col] = 0.5
                elif "_SectorRel" in col or "_vsSPY" in col:
                    df_t[col] = 0.0

            # Fundamentals
            f = funda.get(ticker, {})
            for col in ["market_cap", "pe_ratio", "forward_pe", "ps_ratio",
                        "pb_ratio", "rev_growth", "eps_growth", "current_ratio",
                        "debt_equity", "op_margin", "net_margin", "roe",
                        "fcf_yield", "div_yield", "gross_margin"]:
                df_t[col] = f.get(col, np.nan)

            # Sector one-hot
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
            })
        except Exception as e:
            print(f"  skip {ticker}: {e}")
            continue

    df = pd.DataFrame(results).sort_values("buy_prob", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# =========================================================
# MAIN TRADE LOGIC
# =========================================================

def place_orders(args):
    print("=" * 60)
    print(f"TRADING — {datetime.now().isoformat()}")
    print(f"  TOP_K={args.top_k}  KEEP_N={args.keep_n}  "
          f"$amt={args.dollar_amt:.0f}  dry={DRY_RUN}")
    print("=" * 60)

    # --- Rebalance guard ---
    if not args.no_rebalance:
        state = load_state()
        last_ts = state.get("last_trade_ts")
        if last_ts:
            try:
                last_dt = datetime.fromisoformat(last_ts)
                days_since = (datetime.now() - last_dt).days
                if days_since < args.rebalance_days:
                    print(f"\n  Skipping: last trade was {days_since}d ago "
                          f"(min {args.rebalance_days}d).")
                    print("  Use --no-rebalance to force.")
                    return
            except Exception:
                pass

    # --- 1. Predict ---
    print("\n[1/4] Predicting universe...")
    preds = predict_universe()
    if preds.empty:
        print("No predictions returned.")
        return

    print(f"\nTop {args.top_k} picks:")
    cols = ["rank", "ticker", "price", "buy_prob", "pred"]
    print(preds.head(args.top_k)[cols].to_string())

    keep_set = set(preds.head(args.keep_n)["ticker"])
    buy_set  = set(preds.head(args.top_k)["ticker"])

    # --- 2. Connect ---
    print(f"\n[2/4] Connecting to Alpaca ({'paper' if PAPER else 'LIVE'})...")
    client = get_client()
    current = get_current_positions(client)
    if current:
        print(f"\n  Current positions ({len(current)}): "
              f"{', '.join(f'{k}={v}' for k,v in list(current.items())[:15])}")
    else:
        print(f"\n  No current positions.")

    # --- 3. Compute & place orders ---
    print(f"\n[3/4] {'Simulating' if DRY_RUN else 'Placing'} orders...")

    equity = float(client.get_account().equity)
    max_per_stock = equity * args.max_pct
    per_stock = min(args.dollar_amt, max_per_stock)
    print(f"  Budget per stock: ${per_stock:,.2f}  "
          f"(max {args.max_pct:.0%} of equity = ${max_per_stock:,.2f})")

    actions = []   # log entries

    # --- 3a. SELL: anything held that's NOT in keep_set ---
    # (and not in buy_set — we'll handle those as "keep" anyway)
    for ticker, qty in current.items():
        if ticker in ETF_TICKERS:
            continue   # never touch ETFs
        if ticker in keep_set:
            continue
        # Sell entire position
        print(f"  SELL {ticker}: {qty} shares (not in top {args.keep_n})")
        actions.append({"ticker": ticker, "side": "SELL", "qty": qty,
                        "price": None, "notional": None, "status": "PENDING",
                        "ts": datetime.now().isoformat()})
        if not DRY_RUN:
            try:
                req = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                o = client.submit_order(req)
                print(f"     -> order {o.id} status={o.status}")
                actions[-1]["status"] = str(o.status)
                actions[-1]["order_id"] = str(o.id)
                time.sleep(0.25)
            except Exception as e:
                print(f"     -> FAILED: {e}")
                actions[-1]["status"] = "ERROR"
                actions[-1]["error"] = str(e)
        else:
            actions[-1]["status"] = "DRY_RUN"

    # --- 3b. BUY: top K, skip if already held ---
    for _, row in preds.head(args.top_k).iterrows():
        ticker = row["ticker"]
        price  = float(row["price"])
        if price <= 0:
            print(f"  {ticker}: skipped (price <= 0)")
            continue
        existing = float(current.get(ticker, 0))
        if existing >= 1:
            print(f"  HOLD {ticker}: already own {existing:.0f} shares "
                  f"@ ~${price:.2f}")
            continue
        qty = int(per_stock // price)
        if qty < 1:
            print(f"  {ticker}: skipped (${per_stock:.0f} < 1 share "
                  f"@ ${price:.2f})")
            continue
        notional = qty * price
        print(f"  BUY  {ticker}: {qty} shares @ ~${price:.2f} = ${notional:,.2f}")
        actions.append({"ticker": ticker, "side": "BUY", "qty": qty,
                        "price": price, "notional": notional,
                        "status": "PENDING", "ts": datetime.now().isoformat()})
        if not DRY_RUN:
            try:
                req = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                o = client.submit_order(req)
                print(f"     -> order {o.id} status={o.status}")
                actions[-1]["status"] = str(o.status)
                actions[-1]["order_id"] = str(o.id)
                time.sleep(0.25)
            except Exception as e:
                print(f"     -> FAILED: {e}")
                actions[-1]["status"] = "ERROR"
                actions[-1]["error"] = str(e)
        else:
            actions[-1]["status"] = "DRY_RUN"

    # --- 4. Save log + state ---
    print(f"\n[4/4] Saving log...")
    if actions:
        log_df = pd.DataFrame(actions)
        if os.path.exists(LOG_PATH):
            log_df.to_csv(LOG_PATH, mode='a', header=False, index=False)
        else:
            log_df.to_csv(LOG_PATH, index=False)
        print(f"  Trade log saved: {LOG_PATH}")
        buys  = [a for a in actions if a["side"] == "BUY" and a["notional"]]
        sells = [a for a in actions if a["side"] == "SELL"]
        if buys:
            print(f"  Total BUY notional:  ${sum(a['notional'] for a in buys):,.2f}")
        print(f"  Actions: {len(buys)} buys, {len(sells)} sells")
    else:
        print("  No actions taken.")

    # Save state for rebalance guard
    save_state({"last_trade_ts": datetime.now().isoformat(),
                "last_top_k": list(buy_set),
                "last_keep_n": list(keep_set)})


# =========================================================
# SAFETY: cancel + liquidate
# =========================================================

def cancel_all_open():
    client = get_client()
    try:
        client.cancel_orders()
        print("All open orders cancelled.")
    except Exception as e:
        print(f"Cancel failed: {e}")


def liquidate_all():
    if not PAPER:
        confirm = input("Type 'YES' to liquidate LIVE positions: ")
        if confirm != "YES":
            print("Cancelled.")
            return
    client = get_client()
    try:
        client.close_all_positions(cancel_orders=True)
        print("All positions closed.")
    except Exception as e:
        print(f"Liquidation failed: {e}")


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stonks-XGB trading client (v2.4)")
    parser.add_argument("--mode", default="trade",
                        choices=["trade", "cancel", "liquidate"],
                        help="Action to perform")
    parser.add_argument("--live", action="store_true",
                        help="Use LIVE account (DANGEROUS)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute & log only, no orders submitted")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Number of buys (default {TOP_K})")
    parser.add_argument("--keep-n", type=int, default=KEEP_N,
                        help=f"Hold top N, sell the rest (default {KEEP_N})")
    parser.add_argument("--dollar-amt", type=float, default=DOLLARS_PER_STOCK,
                        help=f"$ per stock (default {DOLLARS_PER_STOCK})")
    parser.add_argument("--max-pct", type=float, default=MAX_POSITION_PCT,
                        help=f"Max %% of equity per position "
                             f"(default {MAX_POSITION_PCT})")
    parser.add_argument("--rebalance-days", type=int, default=REBALANCE_DAYS,
                        help=f"Min days between trades (default {REBALANCE_DAYS})")
    parser.add_argument("--no-rebalance", action="store_true",
                        help="Force trade even if recent run")
    args = parser.parse_args()

    # Flip flags
    if args.live:
        PAPER = False
    if args.dry_run:
        DRY_RUN = True
        print("DRY_RUN enabled — orders will be simulated.")
    else:
        DRY_RUN = False
        if not PAPER:
            confirm = input("Type 'YES' to enable LIVE trading: ")
            if confirm != "YES":
                print("Aborted.")
                sys.exit()

    if args.mode == "trade":
        place_orders(args)
    elif args.mode == "cancel":
        cancel_all_open()
    elif args.mode == "liquidate":
        liquidate_all()

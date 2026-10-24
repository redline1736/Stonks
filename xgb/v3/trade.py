"""
trade.py — v3
=============
Compatible with main.py v3 (FUTURE_DAYS=5, 458 tickers, regime features).

Features:
  --mode trade       : buy top-K, sell anything not in keep-set
  --mode cancel      : cancel all open orders
  --mode liquidate   : close all positions

  --live             : use LIVE account (DANGEROUS, requires confirmation)
  --dry-run          : compute & log only, no orders submitted
  --top-k N          : number of buys (default 10)
  --keep-n N         : rebalance set: hold top N, sell rest (default 20)
  --rebalance-days N : skip trading if last run < N days ago (default 1)
  --no-rebalance     : ignore rebalance-days, always rebalance
  --dollar-amt N     : $ per stock (default 2000)
  --max-pct N        : max % of equity per position (default 0.05)
  --min-prob N       : skip buys with buy_prob < N (default 0.55)
  --sector-cap N     : max positions per sector (default 3)
  --exclude-etf      : skip ETF positions (default: True)
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

ALPACA_API_KEY = "PKIAQ7HSDKYVCDOWKC4KJPDWCL"
ALPACA_SECRET_KEY = "CKyd1cJQtpGvYZ1GVBXK155fKNiWZ8FWfAgwfJ2uvUFe"
PAPER = True   # default: paper trading. Flipped by --live.

TOP_K             = 10
KEEP_N            = 20
DOLLARS_PER_STOCK = 2000.0
MAX_POSITION_PCT  = 0.05
REBALANCE_DAYS    = 1
DRY_RUN           = True    # default safe. CLI flag overrides.
MIN_BUY_PROB      = 0.55
SECTOR_CAP        = 3
EXCLUDE_ETFS      = True

# Model artifacts (must match main.py v3)
MODEL_DIR     = "models"
MODEL_PATH    = os.path.join(MODEL_DIR, "xgboost_model_v3.json")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_list_v3.json")
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler_v3.pkl")
THRESH_PATH   = os.path.join(MODEL_DIR, "threshold_v3.json")
LOG_PATH      = os.path.join(MODEL_DIR, "trade_log_v3.csv")
STATE_PATH    = os.path.join(MODEL_DIR, "trade_state_v3.json")

# Universe (must match main.py v3)
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

ETF_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
    "XLF", "XLE", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB",
    "XLC", "XLRE",
    "EFA", "EEM", "VEA", "VWO", "EWJ", "EWZ", "MCHI", "INDA",
    "TLT", "IEF", "SHY", "BIL", "LQD", "HYG", "JNK", "MUB", "TIP",
    "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PPLT",
    "VNQ", "SCHH", "USRT", "REM",
    "VXX", "UVXY", "SVXY",
    "BITO", "IBIT", "ETHE",
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

# Reuse main.py functions
try:
    from main import (add_features, add_cross_sectional_and_regime,
                      fetch_all_fundamentals, fetch_and_cache,
                      fetch_regime_data, add_regime_features,
                      SECTOR_TICKERS as MAIN_SECTORS,
                      ETF_TICKERS as MAIN_ETFS,
                      REGIME_TICKERS, FUTURE_DAYS)
    HAVE_MAIN = True
except ImportError:
    HAVE_MAIN = False
    print("[warn] Could not import from main.py — falling back to local defs.")


# =========================================================
# STATE
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
# ALPACA
# =========================================================

def get_client():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("ERROR: ALPACA_API_KEY_ID / APCA_API_SECRET_KEY not set.")
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
# MODEL LOADING
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
    for p in sorted(glob.glob(os.path.join(MODEL_DIR, "xgb_seed_*_v3.json"))):
        m = xgb.Booster()
        m.load_model(p)
        ensemble.append(m)
    seen = set(); uniq = []
    for m in ensemble:
        if id(m) not in seen:
            seen.add(id(m)); uniq.append(m)
    return scaler, features, thr, uniq


# =========================================================
# PREDICTION (uses main.py imports)
# =========================================================

def predict_universe(tickers=None):
    if not HAVE_MAIN:
        print("ERROR: cannot run predict without main.py imports.")
        print("  Place trade.py in the same directory as main.py.")
        sys.exit(1)

    scaler, features, thr, ensemble = load_ensemble()
    funda = fetch_all_fundamentals()
    regime = fetch_regime_data()

    if tickers is None:
        if EXCLUDE_ETFS:
            tickers = [t for t in TICKERS if t not in ETF_TICKERS]
        else:
            tickers = TICKERS

    # Pre-compute latest regime values
    regime_latest = {}
    for ticker, rdf in regime.items():
        if rdf.empty:
            continue
        clean = ticker.replace("^", "").replace("-", "_").replace("=", "_")
        latest = rdf.iloc[-1]
        for col in rdf.columns:
            if col == "Date":
                continue
            col_short = col.split("_", 1)[1] if "_" in col else col
            key = f"{clean}_{col_short}"
            v = latest[col]
            regime_latest[key] = float(v) if pd.notna(v) else 0.0

    results = []
    end = datetime.now()
    start = end - timedelta(days=700)  # v3 needs 252d momentum + 5d forward

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
            spy_path = os.path.join(MODEL_DIR, "partial_daily_v3", "SPY.pkl")
            if os.path.exists(spy_path):
                spy = pd.read_pickle(spy_path)
                spy = add_features(spy)
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

            # Cross-sectional placeholders
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
            df_t["Is_ETF"] = 1.0 if ticker in ETF_TICKERS else 0.0

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
                "sector": sector,
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
    print(f"  MIN_PROB={args.min_prob}  SECTOR_CAP={args.sector_cap}  "
          f"EXCLUDE_ETF={EXCLUDE_ETFS}")
    print("=" * 60)

    # Rebalance guard
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

    # 1. Predict
    print("\n[1/4] Predicting universe...")
    preds = predict_universe()
    if preds.empty:
        print("No predictions returned.")
        return

    # Filter by min probability
    pred_buys = preds[preds["buy_prob"] >= args.min_prob].head(args.keep_n)
    print(f"\nTickers above MIN_PROB {args.min_prob}: {len(pred_buys)}")
    print(f"\nTop {args.top_k} picks (above threshold):")
    cols = ["rank", "ticker", "sector", "price", "buy_prob", "pred"]
    print(pred_buys.head(args.top_k)[cols].to_string())

    keep_set = set(pred_buys["ticker"])
    buy_set  = set(pred_buys.head(args.top_k)["ticker"])

    # 2. Connect
    print(f"\n[2/4] Connecting to Alpaca ({'paper' if PAPER else 'LIVE'})...")
    client = get_client()
    current = get_current_positions(client)
    if current:
        # Show non-ETF positions only
        non_etf = {k: v for k, v in current.items() if k not in ETF_TICKERS}
        print(f"\n  Current positions ({len(current)} total, "
              f"{len(non_etf)} non-ETF):")
        for k, v in list(non_etf.items())[:20]:
            print(f"    {k}={v}")
        if len(non_etf) > 20:
            print(f"    ... and {len(non_etf) - 20} more")
    else:
        print(f"\n  No current positions.")

    # 3. Compute & place orders
    print(f"\n[3/4] {'Simulating' if DRY_RUN else 'Placing'} orders...")

    equity = float(client.get_account().equity)
    max_per_stock = equity * args.max_pct
    per_stock = min(args.dollar_amt, max_per_stock)
    print(f"  Budget per stock: ${per_stock:,.2f}  "
          f"(max {args.max_pct:.0%} of equity = ${max_per_stock:,.2f})")

    actions = []

    # Track sector counts for sector-cap
    sector_counts = {}
    # Initialize from current non-ETF positions
    for ticker, qty in current.items():
        if ticker in ETF_TICKERS:
            continue
        s = TICKER_SECTOR.get(ticker, "Unknown")
        sector_counts[s] = sector_counts.get(s, 0) + 1

    # 3a. SELL anything not in keep_set
    for ticker, qty in current.items():
        if ticker in ETF_TICKERS:
            continue
        if ticker in keep_set:
            continue
        print(f"  SELL {ticker}: {qty} shares (not in top {args.keep_n})")
        actions.append({"ticker": ticker, "side": "SELL", "qty": qty,
                        "price": None, "notional": None, "status": "PENDING",
                        "sector": TICKER_SECTOR.get(ticker, "Unknown"),
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
                time.sleep(0.2)
            except Exception as e:
                print(f"     -> FAILED: {e}")
                actions[-1]["status"] = "ERROR"
                actions[-1]["error"] = str(e)
        else:
            actions[-1]["status"] = "DRY_RUN"

    # 3b. BUY top K, with sector cap and skip-if-held
    buy_count = 0
    for _, row in pred_buys.iterrows():
        if buy_count >= args.top_k:
            break
        ticker = row["ticker"]
        sector = row["sector"]
        price  = float(row["price"])
        prob   = float(row["buy_prob"])

        if price <= 0:
            print(f"  {ticker}: skipped (price <= 0)")
            continue
        existing = float(current.get(ticker, 0))
        if existing >= 1:
            print(f"  HOLD {ticker}: already own {existing:.0f} shares "
                  f"@ ~${price:.2f} (prob={prob:.3f})")
            continue

        # Sector cap
        if sector_counts.get(sector, 0) >= args.sector_cap:
            print(f"  {ticker}: skipped (sector {sector} at cap {args.sector_cap})")
            continue

        qty = int(per_stock // price)
        if qty < 1:
            print(f"  {ticker}: skipped (${per_stock:.0f} < 1 share @ ${price:.2f})")
            continue

        notional = qty * price
        print(f"  BUY  {ticker}: {qty} shares @ ~${price:.2f} = ${notional:,.2f}  "
              f"(prob={prob:.3f}, sector={sector})")
        actions.append({
            "ticker": ticker, "side": "BUY", "qty": qty,
            "price": price, "notional": notional,
            "buy_prob": prob,
            "status": "PENDING",
            "sector": sector,
            "ts": datetime.now().isoformat(),
        })
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
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                buy_count += 1
                time.sleep(0.2)
            except Exception as e:
                print(f"     -> FAILED: {e}")
                actions[-1]["status"] = "ERROR"
                actions[-1]["error"] = str(e)
        else:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            buy_count += 1

    # 4. Save log + state
    print(f"\n[4/4] Saving log...")
    if actions:
        log_df = pd.DataFrame(actions)
        if os.path.exists(LOG_PATH):
            log_df.to_csv(LOG_PATH, mode='a', header=False, index=False)
        else:
            log_df.to_csv(LOG_PATH, index=False)
        print(f"  Trade log saved: {LOG_PATH}")

        buys  = [a for a in actions if a["side"] == "BUY" and a.get("notional")]
        sells = [a for a in actions if a["side"] == "SELL"]
        if buys:
            print(f"  Total BUY notional:  ${sum(a['notional'] for a in buys):,.2f}")
            print(f"  Avg buy_prob:        {np.mean([a['buy_prob'] for a in buys]):.3f}")
        if sells:
            print(f"  SELL actions:        {len(sells)}")

        # Sector distribution of buys
        if buys:
            sector_dist = pd.DataFrame(buys)["sector"].value_counts()
            print(f"\n  Sector distribution of buys:")
            print(sector_dist.to_string())
    else:
        print("  No actions taken.")

    save_state({
        "last_trade_ts": datetime.now().isoformat(),
        "last_top_k": list(buy_set),
        "last_keep_n": list(keep_set),
        "last_sector_counts": sector_counts,
    })


# =========================================================
# SAFETY
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
        description="Stonks-XGB v3 trading client")
    parser.add_argument("--mode", default="trade",
                        choices=["trade", "cancel", "liquidate"])
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
                        help=f"Max %% of equity per position (default {MAX_POSITION_PCT})")
    parser.add_argument("--rebalance-days", type=int, default=REBALANCE_DAYS,
                        help=f"Min days between trades (default {REBALANCE_DAYS})")
    parser.add_argument("--no-rebalance", action="store_true",
                        help="Force trade even if recent run")
    parser.add_argument("--min-prob", type=float, default=MIN_BUY_PROB,
                        help=f"Min buy_prob to consider (default {MIN_BUY_PROB})")
    parser.add_argument("--sector-cap", type=int, default=SECTOR_CAP,
                        help=f"Max positions per sector (default {SECTOR_CAP})")
    parser.add_argument("--include-etfs", action="store_true",
                        help="Include ETFs in the universe (default: excluded)")
    args = parser.parse_args()

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
    if args.include_etfs:
        EXCLUDE_ETFS = False

    if args.mode == "trade":
        place_orders(args)
    elif args.mode == "cancel":
        cancel_all_open()
    elif args.mode == "liquidate":
        liquidate_all()

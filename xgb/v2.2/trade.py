"""
Stonks-Alpaca — paper trade top-N model picks
==============================================
- Loads v2.2 model, runs predict_universe
- Buys top N stocks, equal-weight (configurable)
- Places PAPER orders via Alpaca
- Prints order IDs and fills
"""

import os
import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# New alpaca-py SDK
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass

# Reuse your existing model code
from full import (predict_universe, TICKERS, TICKER_SECTOR, SECTOR_NAMES,
                  ETF_TICKERS, SECTOR_TICKERS, add_features,
                  fetch_all_fundamentals, add_cross_sectional_and_regime)

# =========================================================
# CONFIG
# =========================================================

ALPACA_API_KEY = "PKIAQ7HSDKYVCDOWKC4KJPDWCL"
ALPACA_SECRET_KEY = "CKyd1cJQtpGvYZ1GVBXK155fKNiWZ8FWfAgwfJ2uvUFe"
PAPER             = True                                # paper trading

TOP_N             = 5
DOLLARS_PER_STOCK = 1000.0
MAX_POSITION_PCT  = 0.10

MODEL_DIR    = "models"
LOG_PATH     = os.path.join(MODEL_DIR, "trade_log.csv")


# =========================================================
# Client
# =========================================================

def get_client():
    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)
    acct = client.get_account()
    print(f"\nAlpaca account ({'paper' if PAPER else 'LIVE'}):")
    print(f"  Status:        {acct.status}")
    print(f"  Equity:        ${float(acct.equity):,.2f}")
    print(f"  Cash:          ${float(acct.cash):,.2f}")
    print(f"  Buying power:  ${float(acct.buying_power):,.2f}")
    return client


# =========================================================
# Check what we already own
# =========================================================

def get_current_positions(client):
    try:
        positions = client.get_all_positions()
        return {p.symbol: float(p.qty) for p in positions}
    except Exception as e:
        print(f"  Could not fetch positions: {e}")
        return {}


# =========================================================
# Trade
# =========================================================

def place_orders():
    print("=" * 60)
    print(f"TRADING — {datetime.now().isoformat()}")
    print("=" * 60)

    # 1. Get ranked predictions
    print("\n[1/3] Running model on current universe...")
    preds = predict_universe()
    if preds.empty:
        print("No predictions returned.")
        return

    top = preds.head(TOP_N).copy()
    print(f"\nTop {TOP_N} picks:")
    print(top[["rank", "ticker", "price", "buy_prob", "pred"]].to_string())

    # 2. Connect
    print(f"\n[2/3] Connecting to Alpaca ({'paper' if PAPER else 'LIVE'})...")
    client = get_client()

    # Show current positions
    current = get_current_positions(client)
    if current:
        print(f"\n  Current positions ({len(current)}): "
              f"{', '.join(f'{k}={v}' for k,v in list(current.items())[:10])}")
    else:
        print(f"\n  No current positions.")

    # 3. Compute quantities & place orders
    print(f"\n[3/3] Placing orders (DRY_RUN={DRY_RUN})...")
    equity = float(client.get_account().equity)
    max_per_stock = equity * MAX_POSITION_PCT
    per_stock = min(DOLLARS_PER_STOCK, max_per_stock)
    print(f"  Budget per stock: ${per_stock:,.2f}  "
          f"(max {MAX_POSITION_PCT:.0%} of equity = ${max_per_stock:,.2f})")

    orders = []
    for _, row in top.iterrows():
        ticker = row["ticker"]
        price  = float(row["price"])
        if price <= 0:
            print(f"  {ticker}: skipped (price <= 0)")
            continue

        qty = int(per_stock // price)         # whole shares
        if qty < 1:
            print(f"  {ticker}: skipped (${per_stock:.0f} < 1 share @ ${price:.2f})")
            continue

        notional = qty * price
        existing = float(current.get(ticker, 0))
        print(f"  {ticker}: {qty} shares @ ~${price:.2f} = ${notional:,.2f}  "
              f"(already own {existing:.0f})")

        if DRY_RUN:
            orders.append({"ticker": ticker, "qty": qty, "price": price,
                           "notional": notional, "status": "DRY_RUN",
                           "ts": datetime.now().isoformat()})
            continue

        # Build order request
        req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )

        try:
            o = client.submit_order(req)
            orders.append({
                "ticker": ticker, "qty": qty, "price": price,
                "notional": notional, "status": str(o.status),
                "order_id": str(o.id), "ts": datetime.now().isoformat(),
            })
            print(f"     -> order {o.id} status={o.status}")
            time.sleep(0.25)
        except Exception as e:
            print(f"     -> FAILED: {e}")
            orders.append({
                "ticker": ticker, "qty": qty, "price": price,
                "notional": notional, "status": "ERROR",
                "error": str(e), "ts": datetime.now().isoformat(),
            })

    # 4. Save log
    log_df = pd.DataFrame(orders)
    if not log_df.empty:
        if os.path.exists(LOG_PATH):
            log_df.to_csv(LOG_PATH, mode='a', header=False, index=False)
        else:
            log_df.to_csv(LOG_PATH, index=False)
        print(f"\nTrade log saved: {LOG_PATH}")
        print(f"Total notional:   ${log_df['notional'].sum():,.2f}")
    else:
        print("\nNo orders placed.")

    return orders


# =========================================================
# Optional: cancel all open orders (safety)
# =========================================================

def cancel_all_open():
    client = get_client()
    try:
        client.cancel_orders()
        print("All open orders cancelled.")
    except Exception as e:
        print(f"Cancel failed: {e}")


# =========================================================
# Optional: liquidate all positions (panic button)
# =========================================================

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


if __name__ == "__main__":
    DRY_RUN = False
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="trade",
                        choices=["trade", "cancel", "liquidate"])
    parser.add_argument("--dry-run", action="store_true",
                    help="Compute but don't submit (default: LIVE submit)")

    parser.add_argument("--live", action="store_true",
                        help="Use LIVE account (DANGEROUS)")
    args = parser.parse_args()

    if args.dry_run:
        DRY_RUN = True
        print("DRY_RUN enabled — orders will be simulated.")


    if args.live:
        PAPER = False
        if not DRY_RUN:
            confirm = input("Type 'YES' to enable LIVE trading: ")
            if confirm != "YES":
                print("Aborted.")
                exit()

    if args.mode == "trade":
        place_orders()
    elif args.mode == "cancel":
        cancel_all_open()
    elif args.mode == "liquidate":
        liquidate_all()

# =========================================================
# trade_alpaca.py — Execute signals on Alpaca (paper/live)
# Drop-in addition to stonks_xgb_v3.py + stonks_trade.py
# =========================================================

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Dict

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; env vars also work

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError, TimeFrame
except ImportError:
    print("ERROR: pip install alpaca-trade-api")
    sys.exit(1)

# Re-use the signal generator
from stonks_trade import predict_signals, size_positions

# =========================================================
# CONFIG
# =========================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"trade_{datetime.now():%Y%m%d}.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("trade_alpaca")

# Safety: require explicit opt-in for live trading
LIVE_TRADING_ENABLED = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"  # default dry-run

# Order defaults
TIME_IN_FORCE = "day"           # cancels at market close if not filled
EXTENDED_HOURS = False          # keep regular-hours only for predictable fills
POSITION_SIZING = {
    "capital":      float(os.getenv("CAPITAL",      "100000")),
    "risk_per_trade": float(os.getenv("RISK_PCT",   "0.005")),
    "max_positions":  int(os.getenv("MAX_POSITIONS","8")),
}


# =========================================================
# ALPACA CLIENT
# =========================================================

def get_api():
    key      = os.getenv("APCA_API_KEY_ID")
    secret   = os.getenv("APCA_API_SECRET_KEY")
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    if not key or not secret:
        raise RuntimeError(
            "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in env or .env file"
        )

    api = tradeapi.REST(key, secret, base_url, api_version="v2")
    return api, base_url


def detect_environment(base_url: str) -> str:
    if "paper" in base_url:
        return "PAPER"
    return "LIVE"


def account_summary(api):
    acct = api.get_account()
    return {
        "equity":         float(acct.equity),
        "cash":           float(acct.cash),
        "buying_power":   float(acct.buying_power),
        "daytrade_count": int(acct.daytrade_count),
        "pattern_day_trader": acct.pattern_day_trader,
        "trading_blocked": acct.trading_blocked,
        "account_blocked":  acct.account_blocked,
    }


def market_state(api) -> Dict:
    clock = api.get_clock()
    return {
        "is_open":        clock.is_open,
        "next_open":      clock.next_open,
        "next_close":     clock.next_close,
        "timestamp":      clock.timestamp,
    }


# =========================================================
# ORDER PLACEMENT
# =========================================================

def place_bracket_order(
    api,
    ticker: str,
    qty: int,
    take_profit_price: float,
    stop_loss_price: float,
    side: str = "buy",
    time_in_force: str = TIME_IN_FORCE,
):
    """
    Place a single bracket order (entry + TP + SL atomically).
    Entry is a market order; SL is a stop-market; TP is a limit.
    """
    if qty <= 0:
        log.warning(f"{ticker}: qty={qty} <= 0, skipping")
        return None

    # Price sanity: round to 2 decimals, ensure TP > entry > SL on buys
    tp = round(float(take_profit_price), 2)
    sl = round(float(stop_loss_price), 2)

    if side == "buy":
        if tp <= sl:
            log.warning(f"{ticker}: TP {tp} <= SL {sl}, skipping")
            return None
    else:
        if tp >= sl:
            log.warning(f"{ticker}: short TP {tp} >= SL {sl}, skipping")
            return None

    order_payload = {
        "symbol":        ticker,
        "qty":           str(qty),
        "side":          side,
        "type":          "market",
        "time_in_force": time_in_force,
        "order_class":   "bracket",
        "take_profit":   {"limit_price": f"{tp:.2f}"},
        "stop_loss":     {"stop_price":  f"{sl:.2f}"},
    }

    if EXTENDED_HOURS:
        order_payload["extended_hours"] = True

    if DRY_RUN or not LIVE_TRADING_ENABLED:
        log.info(f"[DRY-RUN] {ticker} {side} x{qty} | TP={tp} SL={sl}")
        return {"dry_run": True, "payload": order_payload}

    try:
        order = api.submit_order(**order_payload)
        log.info(
            f"[LIVE] Submitted bracket {ticker} {side} x{qty} "
            f"entry=market TP={tp} SL={sl} | order_id={order.id}"
        )
        return order._raw if hasattr(order, "_raw") else vars(order)
    except APIError as e:
        log.error(f"[LIVE] {ticker} order FAILED: {e}")
        return None


def cancel_all_orders(api, ticker: Optional[str] = None):
    try:
        if ticker:
            api.cancel_orders()
        else:
            api.cancel_all_orders()
        log.info(f"Cancelled orders{' for ' + ticker if ticker else ''}")
    except APIError as e:
        log.error(f"Cancel failed: {e}")


def close_position(api, ticker: str):
    if DRY_RUN or not LIVE_TRADING_ENABLED:
        log.info(f"[DRY-RUN] would close {ticker}")
        return
    try:
        api.close_position(ticker)
        log.info(f"Closed position: {ticker}")
    except APIError as e:
        log.error(f"Close {ticker} failed: {e}")


def close_all_positions(api):
    if DRY_RUN or not LIVE_TRADING_ENABLED:
        log.info("[DRY-RUN] would close all positions")
        return
    try:
        api.close_all_positions()
        log.info("Closed all positions")
    except APIError as e:
        log.error(f"Close-all failed: {e}")


# =========================================================
# MAIN EXECUTION
# =========================================================

def run_daily(force: bool = False):
    """
    Daily entry:
      1. Check market state
      2. Generate signals
      3. Size positions
      4. Place bracket orders
    """
    api, base_url = get_api()
    env = detect_environment(base_url)

    if env == "LIVE" and not LIVE_TRADING_ENABLED:
        log.error("=" * 60)
        log.error("LIVE TRADING BLOCKED — set ENABLE_LIVE_TRADING=true to allow")
        log.error("=" * 60)
        return
    if DRY_RUN and env == "LIVE":
        log.warning("DRY_RUN=true on LIVE env — orders will NOT be placed")

    log.info("=" * 60)
    log.info(f"Alpaca environment: {env}  DRY_RUN={DRY_RUN}")
    log.info("=" * 60)

    # Account state
    try:
        acct = account_summary(api)
        log.info(f"Account: equity=${acct['equity']:,.2f}  "
                 f"cash=${acct['cash']:,.2f}  "
                 f"buying_power=${acct['buying_power']:,.2f}")
        if acct["trading_blocked"] or acct["account_blocked"]:
            log.error(f"Account is blocked: {acct}")
            return
    except APIError as e:
        log.error(f"Cannot read account: {e}")
        return

    # Market state
    try:
        mkt = market_state(api)
        log.info(f"Market open: {mkt['is_open']}  "
                 f"next_open={mkt['next_open']}  "
                 f"next_close={mkt['next_close']}")
    except APIError as e:
        log.error(f"Cannot read market clock: {e}")
        return

    if not mkt["is_open"] and not force:
        log.warning("Market is closed. Use --force to override (orders will queue).")

    # Get existing positions to avoid double-buying
    try:
        existing = {p.symbol: int(float(p.qty))
                    for p in api.list_positions() if float(p.qty) != 0}
    except APIError as e:
        log.error(f"Cannot read positions: {e}")
        return

    log.info(f"Existing positions: {existing or 'none'}")

    # Generate signals
    log.info("Generating signals...")
    signals = predict_signals()
    if signals.empty:
        log.warning("No signals generated")
        return

    # Size positions
    sized = size_positions(
        signals,
        capital=POSITION_SIZING["capital"],
        risk_per_trade=POSITION_SIZING["risk_per_trade"],
        max_positions=POSITION_SIZING["max_positions"],
    )
    if sized.empty:
        log.warning("No positions to take after sizing")
        return

    # Buying-power check: cap total notional at 80% of buying power
    total_notional = sized["position_value"].sum()
    bp_buffer      = 0.80
    if total_notional > acct["buying_power"] * bp_buffer:
        log.warning(
            f"Total notional ${total_notional:,.0f} exceeds "
            f"{bp_buffer:.0%} of buying power (${acct['buying_power']*bp_buffer:,.0f}). "
            f"Scaling down."
        )
        scale = (acct["buying_power"] * bp_buffer) / total_notional
        sized["shares"] = (sized["shares"] * scale).astype(int)
        sized = sized[sized["shares"] > 0]
        sized["position_value"] = sized["shares"] * sized["entry"]
        sized["max_loss_dollars"] = sized["shares"] * sized["risk_per_share"]
        sized["target_gain_dollars"] = sized["shares"] * sized["reward_per_share"]

    # Place orders
    log.info("=" * 60)
    log.info("PLACING ORDERS")
    log.info("=" * 60)

    orders_placed = []
    skipped = []

    for _, row in sized.iterrows():
        t = row["ticker"]
        if t in existing:
            log.info(f"{t}: already held ({existing[t]} shares), skipping")
            skipped.append(t)
            continue

        result = place_bracket_order(
            api,
            ticker=t,
            qty=int(row["shares"]),
            take_profit_price=row["take_profit"],
            stop_loss_price=row["stop_loss"],
        )
        if result is not None:
            orders_placed.append(t)

    # Save execution log
    exec_log = {
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "env":             env,
        "dry_run":         DRY_RUN,
        "account_equity":  acct["equity"],
        "buying_power":    acct["buying_power"],
        "existing":        existing,
        "orders_placed":   orders_placed,
        "skipped":         skipped,
        "signals_count":   len(signals),
        "sized_count":     len(sized),
    }
    log_path = os.path.join(LOG_DIR, f"exec_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(log_path, "w") as f:
        json.dump(exec_log, f, indent=2)
    log.info(f"Execution log: {log_path}")

    # Summary
    log.info("=" * 60)
    log.info(f"SUMMARY: {len(orders_placed)} orders placed, "
             f"{len(skipped)} skipped, {len(sized) - len(orders_placed) - len(skipped)} failed")
    log.info("=" * 60)


def flatten_all():
    """Emergency: cancel all orders + close all positions."""
    api, base_url = get_api()
    env = detect_environment(base_url)
    log.warning(f"FLATTEN_ALL on {env}")
    cancel_all_orders(api)
    close_all_positions(api)


def show_status():
    """Print account, positions, and open orders."""
    api, base_url = get_api()
    env = detect_environment(base_url)
    print(f"\n=== Alpaca {env} ===")

    try:
        acct = account_summary(api)
        print(f"\nAccount:")
        for k, v in acct.items():
            print(f"  {k}: {v}")
    except APIError as e:
        print(f"Account error: {e}")

    try:
        mkt = market_state(api)
        print(f"\nMarket: open={mkt['is_open']}")
    except APIError as e:
        print(f"Clock error: {e}")

    try:
        positions = api.list_positions()
        if positions:
            print(f"\nPositions ({len(positions)}):")
            for p in positions:
                pnl = float(p.unrealized_pl)
                pnl_pct = float(p.unrealized_plpc) * 100
                print(f"  {p.symbol:>6} qty={p.qty:>6} "
                      f"entry=${float(p.avg_entry_price):>7.2f} "
                      f"now=${float(p.current_price):>7.2f} "
                      f"pnl=${pnl:>+8.2f} ({pnl_pct:>+6.2f}%)")
        else:
            print("\nNo open positions")
    except APIError as e:
        print(f"Positions error: {e}")

    try:
        orders = api.list_orders(status="open")
        if orders:
            print(f"\nOpen orders ({len(orders)}):")
            for o in orders:
                print(f"  {o.symbol:>6} {o.side:>4} {o.type:>10} "
                      f"qty={o.qty} status={o.status}")
        else:
            print("\nNo open orders")
    except APIError as e:
        print(f"Orders error: {e}")


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode",
                   choices=["run", "status", "flatten", "dry-run"],
                   default="dry-run")
    p.add_argument("--force", action="store_true",
                   help="Run even if market is closed")
    args = p.parse_args()

    if args.mode == "run":
        # Override DRY_RUN if user passes --mode run
        DRY_RUN_OVERRIDE = os.getenv("DRY_RUN_OVERRIDE", "false").lower() == "true"
        if DRY_RUN_OVERRIDE:
            DRY_RUN = False
        run_daily(force=args.force)

    elif args.mode == "dry-run":
        DRY_RUN = True
        run_daily(force=args.force)

    elif args.mode == "status":
        show_status()

    elif args.mode == "flatten":
        confirm = input("Type 'YES' to flatten all positions and orders: ")
        if confirm == "YES":
            flatten_all()
        else:
            print("Cancelled")


# Model.py
** 1. Make sure you've already run:**
python stonks_xgb_v3.py --mode build
** 2. Train the quantile models**
- python stonks_trade.py --mode train
- Every morning, 5 min before open:

- python stonks_trade.py --mode signal
- You get a CSV with BUY/SKIP per ticker, plus entry / TP / SL prices. The pred_med_from_entry column is the model's median expected return from your actual entry price (gap-adjusted). The confidence column is |median| / (q75 - q25) — a tight distribution = high confidence.

**To size positions:**

- python stonks_trade.py --mode size --capital 100000 --risk 0.005 --max-pos 8

**1. Build the dataset (downloads all data, ~15-30 min first run)**
- python stonks_xgb_v3.py --mode build

**2. Train the classifier model**
- python stonks_xgb_v3.py --mode train

**3. Train the quantile models (needed for the trading system)**
- python stonks_trade.py --mode train


# Run ~5 min before market open (9:25 AM ET)

**A. Get signals only (no sizing)**
- python stonks_trade.py --mode signal

**B. Get signals with position sizing**
- python stonks_trade.py --mode size --capital 100000 --risk 0.005 --max-pos 8

**Re-run prediction only (uses existing v3 classifier, no retraining)**
- python stonks_xgb_v3.py --mode predict

**Run the fundamentals screen (no ML, just PE/margins filter)**
- python stonks_xgb_v3.py --mode screen

**Smaller account, tighter risk**
- python stonks_trade.py --mode size --capital 25000 --risk 0.003 --max-pos 5

**Larger account, more positions**
- python stonks_trade.py --mode size --capital 500000 --risk 0.004 --max-pos 12

**Aggressive (higher risk per trade)**
- python stonks_trade.py --mode size --capital 100000 --risk 0.01 --max-pos 6

# Trade.py

**Check account status (no orders placed)**
- python trade_alpaca.py --mode status

**Dry run (default; see what would happen, no orders)**
- python trade_alpaca.py --mode dry-run

** Place orders on paper account**
- DRY_RUN_OVERRIDE=true python trade_alpaca.py --mode run

**Place orders on LIVE account (requires ENABLE_LIVE_TRADING=true)**
- ENABLE_LIVE_TRADING=true DRY_RUN_OVERRIDE=true python trade_alpaca.py --mode run

**Force-run on a weekend (orders will queue)**
- python trade_alpaca.py --mode run --force

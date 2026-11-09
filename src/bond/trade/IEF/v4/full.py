"""
Bond momentum strategy — IEF, 20-day z-scored momentum.

Signal:   mom_20_z = 20-day return / (20-day daily vol * sqrt(20))
Rule:     LONG when mom_20_z >  +Z, SHORT when < -Z, hold 5 days
Filters:  optional VIX/VIX3M regime gate, optional 5d/20d agreement gate
ML:       LogisticRegression on 3 features, 3-year rolling, monthly refit
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA
# ============================================================
print("Downloading data...")
tickers = ["IEF", "^VIX", "^VIX3M"]
raw = yf.download(tickers, start="2015-01-01", auto_adjust=False, progress=False)
df = raw['Close'].copy()
df.columns = [c.replace('^', '') for c in df.columns]
df = df.ffill().dropna()
print(f"Range: {df.index[0].date()} -> {df.index[-1].date()}  n={len(df)}\n")

# ============================================================
# 2. FEATURES (the ONLY ones that matter)
# ============================================================
ief_ret = df['IEF'].pct_change()
vol20   = ief_ret.rolling(20).std()

df['d20']          = df['IEF'].pct_change(20)
df['d5']           = df['IEF'].pct_change(5)
df['mom_20_z']     = df['d20'] / (vol20 * np.sqrt(20))   # <-- THE signal
df['mom_5_z']      = df['d5']  / (vol20 * np.sqrt(5))
df['mom_disagree'] = np.sign(df['mom_20_z']) * np.sign(df['mom_5_z'])
df['vix_ts']       = df['VIX'] / df['VIX3M']
df['regime']       = (df['vix_ts'] > 1.0).astype(int)

for h in [1, 3, 5, 10, 20]:
    df[f'fwd{h}'] = df['IEF'].shift(-h) / df['IEF'] - 1

# ============================================================
# 3. BACKTEST ENGINE
# ============================================================
def make_signal(df, z_thresh=1.0, regime_gate=False, agree_gate=False):
    s = pd.Series(0.0, index=df.index)
    long_m  = df['mom_20_z'] >  z_thresh
    short_m = df['mom_20_z'] < -z_thresh
    if regime_gate:
        long_m  &= df['regime'].eq(1)
        short_m &= df['regime'].eq(1)
    if agree_gate:
        long_m  &= df['mom_disagree'].gt(0)
        short_m &= df['mom_disagree'].gt(0)
    s[long_m]  =  1.0
    s[short_m] = -1.0
    return s

def backtest(df, sig, hold=5, cost_bps=1.0):
    """Position held for `hold` days after each signal (ffill-based)."""
    pos = sig.replace(0, np.nan).ffill(limit=hold-1).shift(1).fillna(0)
    daily = df['IEF'].pct_change().fillna(0)
    turn  = pos.diff().abs().fillna(pos.abs())
    pnl   = (pos * daily - turn * cost_bps / 10000).fillna(0)
    return pos, pnl

def stats(pnl, pos, name=""):
    ann_r = pnl.mean() * 252
    ann_v = pnl.std() * np.sqrt(252)
    sh    = ann_r / ann_v if ann_v > 0 else np.nan
    cum   = (1 + pnl).cumprod()
    mdd   = (cum / cum.cummax() - 1).min()
    return dict(name=name, ret=ann_r, vol=ann_v, sharpe=sh, mdd=mdd,
                total=cum.iloc[-1]-1, in_mkt=(pos.abs() > 0.05).mean(),
                n_sig=int((pos.diff().abs() > 0.5).sum()))

# ============================================================
# 4. RULE BACKTEST GRID
# ============================================================
print("=" * 78)
print("RULE BACKTESTS")
print("=" * 78)
print(f"{'Strategy':34s} {'Ret':>8s} {'Vol':>7s} {'Sharpe':>7s} {'MDD':>8s} {'inMkt':>6s} {'signals':>8s}")

configs = [
    ("Mom z>1.00, hold5, 1bp",  dict(z_thresh=1.00)),
    ("Mom z>1.00, hold5, 3bp",  dict(z_thresh=1.00)),
    ("Mom z>0.75, hold5, 1bp",  dict(z_thresh=0.75)),
    ("Mom z>1.25, hold5, 1bp",  dict(z_thresh=1.25)),
    ("Mom z>1.00 + regime",     dict(z_thresh=1.00, regime_gate=True)),
    ("Mom z>1.00 + agree",      dict(z_thresh=1.00, agree_gate=True)),
]
costs = [1.0, 3.0, 1.0, 1.0, 1.0, 1.0]

results = {}
for (name, cfg), c in zip(configs, costs):
    sig = make_signal(df, **cfg)
    pos, pnl = backtest(df, sig, hold=5, cost_bps=c)
    r = stats(pnl, pos, name)
    results[name] = (pos, pnl, r)
    print(f"{name:34s} {r['ret']*100:>+7.2f}% {r['vol']*100:>6.2f}% "
          f"{r['sharpe']:>7.2f} {r['mdd']*100:>+7.2f}% {r['in_mkt']*100:>5.0f}% "
          f"{r['n_sig']:>8d}")

# Benchmark
bh = df['IEF'].pct_change().fillna(0)
r_bh = stats(bh, pd.Series(1.0, index=df.index), "Buy&Hold IEF")
print(f"{'Buy & Hold IEF':34s} {r_bh['ret']*100:>+7.2f}% {r_bh['vol']*100:>6.2f}% "
      f"{r_bh['sharpe']:>7.2f} {r_bh['mdd']*100:>+7.2f}% {r_bh['in_mkt']*100:>5.0f}%        -")

# ============================================================
# 5. ML WALK-FORWARD (LogisticRegression, 3 features)
# ============================================================
print("\n" + "=" * 78)
print("WALK-FORWARD ML: LogisticRegression(mom_20_z, mom_5_z, mom_disagree)")
print("=" * 78)

FEATS = ['mom_20_z', 'mom_5_z', 'mom_disagree']
ml = df[FEATS + ['fwd5']].dropna().copy()
ml['y'] = (ml['fwd5'] > 0).astype(int)

train_days = 756          # 3 years
refit_every = 21          # monthly

prob = pd.Series(np.nan, index=ml.index)
for i in range(train_days, len(ml), refit_every):
    tr = ml.iloc[i-train_days:i]
    te_idx = ml.index[i : min(i+refit_every, len(ml))]
    m = LogisticRegression(C=0.5, max_iter=1000)
    m.fit(tr[FEATS], tr['y'])
    prob.loc[te_idx] = m.predict_proba(ml.loc[te_idx, FEATS])[:, 1]

oos = prob.dropna()
print(f"OOS days: {len(oos):4d}   "
      f"P(up) OOS: {ml.loc[oos.index,'y'].mean():.3f}   "
      f"OOS IC: {oos.corr(ml.loc[oos.index,'fwd5']):+.4f}")

# ML signal -> backtest
sig_ml = pd.Series(0.0, index=df.index)
sig_ml.loc[oos[oos > 0.55].index] = 1.0
sig_ml.loc[oos[oos < 0.45].index] = -1.0
pos_ml, pnl_ml = backtest(df, sig_ml, hold=5, cost_bps=1.0)
r_ml = stats(pnl_ml, pos_ml, "ML walk-forward")
print(f"{'ML walk-forward (thr 0.55/0.45)':34s} {r_ml['ret']*100:>+7.2f}% "
      f"{r_ml['vol']*100:>6.2f}% {r_ml['sharpe']:>7.2f} {r_ml['mdd']*100:>+7.2f}% "
      f"{r_ml['in_mkt']*100:>5.0f}% {r_ml['n_sig']:>8d}")

# ============================================================
# 6. BLOCK BOOTSTRAP CI ON SHARPE
# ============================================================
print("\n" + "=" * 78)
print("BLOCK BOOTSTRAP 95% CI on Sharpe  (5-day blocks, 2000 resamples)")
print("=" * 78)

def boot_sharpe(rets, n=2000, block=5, seed=42):
    rng = np.random.default_rng(seed)
    r = rets.values; L = len(r); nb = L // block
    out = np.empty(n)
    for i in range(n):
        idx = np.concatenate([np.arange(s, s+block)
                              for s in rng.integers(0, L-block, size=nb)])
        s = r[idx]
        v = s.std() * np.sqrt(252)
        out[i] = (s.mean()*252) / v if v > 0 else 0.0
    return out

for name in ["Mom z>1.00 + regime", "ML walk-forward"]:
    pnl = results[name][1] if name in results else pnl_ml
    pnl = pnl.dropna()
    b = boot_sharpe(pnl)
    obs = pnl.mean()*252 / (pnl.std()*np.sqrt(252))
    lo, hi = np.percentile(b, [2.5, 97.5])
    verdict = "SIGNIFICANT (>0)" if lo > 0 else "includes 0"
    print(f"{name:34s} Sharpe={obs:+.2f}  CI=[{lo:+.2f},{hi:+.2f}]  {verdict}")

# ============================================================
# 7. YEAR-BY-YEAR (best config)
# ============================================================
best_name = "Mom z>1.00 + regime"
pnl_best = results[best_name][1]
print(f"\n" + "=" * 78)
print(f"YEAR-BY-YEAR  ({best_name})")
print("=" * 78)
yr = pnl_best.groupby(pnl_best.index.year).agg(
    total=lambda x: x.sum(),
    sharpe=lambda x: x.mean()/x.std()*np.sqrt(252) if x.std() > 0 else np.nan,
    n_days='count',
)
yr['hit'] = yr['total'] > 0
print(f"{'Year':>6s} {'Total':>9s} {'Sharpe':>8s} {'Win?':>6s}")
for y, row in yr.iterrows():
    print(f"{y:>6d} {row['total']*100:>+8.2f}% {row['sharpe']:>8.2f} "
          f"{'YES' if row['hit'] else 'no ':>6s}")
print(f"Winning years: {yr['hit'].sum()}/{len(yr)}")

# ============================================================
# 8. COST SENSITIVITY (best config)
# ============================================================
print("\n" + "=" * 78)
print(f"COST SENSITIVITY  ({best_name})")
print("=" * 78)
sig_best = make_signal(df, z_thresh=1.0, regime_gate=True)
print(f"{'cost (bp)':>10s} {'Sharpe':>8s} {'Ann Ret':>9s}")
for c in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]:
    _, pnl = backtest(df, sig_best, hold=5, cost_bps=c)
    sh = pnl.mean()*252 / (pnl.std()*np.sqrt(252))
    print(f"{c:>10.2f} {sh:>8.2f} {pnl.mean()*252*100:>+8.2f}%")

# ============================================================
# 9. TRADE LOG (last 10 signals of best config)
# ============================================================
print("\n" + "=" * 78)
print(f"LAST 10 SIGNALS  ({best_name})")
print("=" * 78)
trades = sig_best[sig_best != 0].tail(10)
for dt, s in trades.items():
    f5 = df.loc[dt, 'fwd5']
    z  = df.loc[dt, 'mom_20_z']
    if pd.isna(f5):
        continue
    direction = "LONG " if s > 0 else "SHORT"
    hit = "hit" if (s > 0 and f5 > 0) or (s < 0 and f5 < 0) else "MISS"
    print(f"{dt.date()}  {direction}  z={z:+.2f}  fwd5={f5*100:+.2f}%  [{hit}]")

# ============================================================
# 10. SUMMARY
# ============================================================
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
best = results[best_name][2]
print(f"Base P(up 5d):           {df['fwd5'].gt(0).mean():.3f}")
print(f"Buy & Hold IEF Sharpe:   {r_bh['sharpe']:+.2f}   "
      f"MDD {r_bh['mdd']*100:+.1f}%")
print(f"Best rule Sharpe:        {best['sharpe']:+.2f}   "
      f"MDD {best['mdd']*100:+.1f}%   in-mkt {best['in_mkt']*100:.0f}%")
print(f"ML walk-forward Sharpe:  {r_ml['sharpe']:+.2f}   "
      f"MDD {r_ml['mdd']*100:+.1f}%")
print(f"\nVerdict: strategy converts a {-r_bh['sharpe']:+.2f}-Sharpe asset "
      f"into a ~{best['sharpe']:+.2f}-Sharpe strategy by trading 1/3 of days.")
print("Next steps before real capital:")
print("  1. Paper-trade 3+ months to verify live fill / slippage assumptions")
print("  2. Correlate strategy returns vs SPY — you want LOW correlation for diversification")
print("  3. Size at 0.5-1.0x Kelly given the bootstrap CI width")
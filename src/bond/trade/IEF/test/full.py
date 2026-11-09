"""
Fixed diagnostic suite — completes all 10 tests from where you left off.
Includes: perm, DSR, long/short, Kelly, corr, serial, regime, leave-out.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings; warnings.filterwarnings('ignore')

# ---------- DATA + FEATURES (unchanged) ----------
tickers = ["IEF", "^VIX", "^VIX3M", "SPY", "TLT"]
raw = yf.download(tickers, start="2015-01-01", auto_adjust=False, progress=False)
df = raw['Close'].copy()
df.columns = [c.replace('^', '') for c in df.columns]
df = df.ffill().dropna()

ief_ret = df['IEF'].pct_change()
vol20   = ief_ret.rolling(20).std()
df['mom_20_z'] = df['IEF'].pct_change(20) / (vol20 * np.sqrt(20))
df['mom_5_z']  = df['IEF'].pct_change(5)  / (vol20 * np.sqrt(5))
df['vix_ts']   = df['VIX'] / df['VIX3M']
df['regime']   = (df['vix_ts'] > 1.0).astype(int)
for h in [1, 5, 10]:
    df[f'fwd{h}'] = df['IEF'].shift(-h) / df['IEF'] - 1

def make_signal(df, z=1.25, regime=False):
    s = pd.Series(0.0, index=df.index)
    L = df['mom_20_z'] > z
    S = df['mom_20_z'] < -z
    if regime:
        L &= df['regime'].eq(1); S &= df['regime'].eq(1)
    s[L] = 1.0; s[S] = -1.0
    return s

def backtest(df, sig, hold=5, cost_bps=1.0):
    pos = sig.replace(0, np.nan).ffill(limit=hold-1).shift(1).fillna(0)
    daily = df['IEF'].pct_change().fillna(0)
    turn = pos.diff().abs().fillna(pos.abs())
    pnl = (pos * daily - turn * cost_bps/10000).fillna(0)
    return pos, pnl

def sharpe(r):
    return r.mean()*252 / (r.std()*np.sqrt(252)) if r.std()>0 else np.nan

sig = make_signal(df, z=1.25)
pos, pnl = backtest(df, sig, hold=5, cost_bps=1.0)

# ============================================================
# TEST B — FIXED: rolling 3y Sharpe
# ============================================================
print("="*80); print("TEST B: ROLLING 3-YEAR SHARPE"); print("="*80)
roll_sh = pnl.rolling(756).apply(lambda x: sharpe(pd.Series(x)) if len(x)>100 else np.nan)
# FIX: resample the Series itself, don't index original by quarter-end dates
roll_q = roll_sh.resample('QE').last().dropna()
for dt, v in roll_q.items():
    print(f"  {dt.date()}  3y-Sharpe = {v:+.2f}")

# ============================================================
# TEST C — permutation
# ============================================================
print("\n" + "="*80); print("TEST C: PERMUTATION (shuffle signal timing, 1000x)"); print("="*80)
rng = np.random.default_rng(42)
real_sh = sharpe(pnl)
sig_arr = sig.values.copy()
sig_idx = np.where(sig_arr != 0)[0]
null = np.empty(1000)
for i in range(1000):
    shuf = np.zeros_like(sig_arr)
    shuf[rng.permutation(sig_idx)] = sig_arr[sig_idx]
    _, p2 = backtest(df, pd.Series(shuf, index=df.index), hold=5, cost_bps=1.0)
    null[i] = sharpe(p2) if p2.std()>0 else 0
pval = (null >= real_sh).mean()
print(f"Real Sharpe: {real_sh:+.3f}")
print(f"Null mean: {null.mean():+.3f}  std: {null.std():.3f}  95th pct: {np.percentile(null,95):+.2f}")
print(f"p-value (one-sided): {pval:.3f}  -> {'SIGNIFICANT' if pval<0.05 else 'NOT significant'}")

# ============================================================
# TEST D — deflated Sharpe
# ============================================================
print("\n" + "="*80); print("TEST D: DEFLATED SHARPE (corrects for 20 configs tested)"); print("="*80)
n_tests = 20
T = len(pnl.dropna())
skew = pnl.skew(); kurt = pnl.kurt()
e_max = norm.ppf(1 - 1/n_tests) / np.sqrt(252)
sr = sharpe(pnl)
denom = np.sqrt(max(1 - skew*sr + (kurt-1)/4*sr*sr, 1e-9))
dsr = (sr - e_max) * np.sqrt(T/252) / denom
print(f"Expected max Sharpe from {n_tests} null trials: {e_max:+.3f}")
print(f"Observed Sharpe: {sr:+.3f}")
print(f"Deflated SR t-stat: {dsr:+.2f}  -> {'survives' if dsr>2 else 'DOES NOT survive'} multiple testing")

# ============================================================
# TEST E — long vs short asymmetry
# ============================================================
print("\n" + "="*80); print("TEST E: LONG vs SHORT ASYMMETRY"); print("="*80)
ld = pnl[pos > 0.5]; sd = pnl[pos < -0.5]
for lbl, series in [("Long", ld), ("Short", sd)]:
    if len(series) < 20: print(f"{lbl}: too few ({len(series)})"); continue
    print(f"{lbl:6s} n={len(series):4d}  Sharpe={sharpe(series):+.2f}  "
          f"ret={series.sum()*100:+.2f}%  hit={(series[series!=0]>0).mean():.1%}")

# ============================================================
# TEST F — trade-level + Kelly
# ============================================================
print("\n" + "="*80); print("TEST F: TRADE-LEVEL STATS + HALF-KELLY"); print("="*80)
tr = []
for dt in sig[sig!=0].index:
    if dt in df.index and pd.notna(df.loc[dt, 'fwd5']):
        tr.append(sig.loc[dt] * df.loc[dt, 'fwd5'] - 2/10000)
tr = np.array(tr)
if len(tr) > 10:
    w, l = tr[tr>0], tr[tr<0]
    wr = (tr>0).mean()
    payoff = w.mean()/abs(l.mean()) if len(l) and l.mean()!=0 else np.nan
    kelly = wr - (1-wr)/payoff if payoff>0 else np.nan
    t = tr.mean()/(tr.std()/np.sqrt(len(tr)))
    print(f"N={len(tr)}  win={wr:.2%}  payoff={payoff:.2f}")
    print(f"Half-Kelly: {kelly/2:+.2%}   trade mean: {tr.mean()*100:+.3f}%  "
          f"t-stat: {t:+.2f}")

# ============================================================
# TEST G — correlations
# ============================================================
print("\n" + "="*80); print("TEST G: CORRELATIONS"); print("="*80)
spy = df['SPY'].pct_change().reindex(pnl.index).fillna(0)
tlt = df['TLT'].pct_change().reindex(pnl.index).fillna(0)
ief = ief_ret.reindex(pnl.index).fillna(0)
print(f"corr(pnl, SPY)   = {pnl.corr(spy):+.3f}")
print(f"corr(pnl, TLT)   = {pnl.corr(tlt):+.3f}")
print(f"corr(pnl, IEF)   = {pnl.corr(ief):+.3f}")
print(f"corr(pnl, |IEF|) = {pnl.corr(ief.abs()):+.3f}  <- if >0.3, capturing vol not direction")

# ============================================================
# TEST H — serial correlation / effective N
# ============================================================
print("\n" + "="*80); print("TEST H: SERIAL CORRELATION"); print("="*80)
ac1 = pnl.autocorr(1)
n_eff = len(pnl) * (1-ac1)/(1+ac1) if abs(ac1)<1 else len(pnl)
print(f"AC(1)={ac1:+.3f}  Effective N≈{n_eff:.0f}/{len(pnl)}")

# ============================================================
# TEST I — regime breakdown
# ============================================================
print("\n" + "="*80); print("TEST I: REGIME BREAKDOWN"); print("="*80)
regimes = {
    "VIX TS > 1.0 (backwardation)": df['vix_ts'] > 1.0,
    "VIX TS < 0.90 (contango)":     df['vix_ts'] < 0.90,
    "VIX 0.90–1.00 (transition)":   (df['vix_ts']>=0.90)&(df['vix_ts']<=1.00),
}
print(f"{'Regime':35s} {'n':>5s} {'Sharpe':>8s} {'ret':>9s}")
for name, mask in regimes.items():
    m = mask.reindex(pnl.index).fillna(False)
    p = pnl[m]
    if len(p) > 30:
        print(f"{name:35s} {len(p):5d} {sharpe(p):>+8.2f} {p.sum()*100:>+8.2f}%")

# ============================================================
# TEST J — leave-one-year-out
# ============================================================
print("\n" + "="*80); print("TEST J: LEAVE-ONE-YEAR-OUT"); print("="*80)
base_sh = sharpe(pnl)
print(f"Baseline Sharpe: {base_sh:+.2f}")
for y in sorted(pnl.index.year.unique()):
    if y == pnl.index.year.max(): continue
    p = pnl[pnl.index.year != y]
    sh = sharpe(p); delta = sh - base_sh
    flag = " *** CARRY" if delta < -0.15 else ""
    print(f"  drop {y}: Sharpe={sh:+.2f}  Δ={delta:+.2f}{flag}")
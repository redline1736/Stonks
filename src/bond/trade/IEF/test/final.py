"""
Final validation suite — IEF bond momentum.

Variants:
  V0  Baseline     z>1.25, 5-day hold, no filter (from v4)
  V1  Binary gate  v4 regime filter (enter only if VIX/VIX3M > 1.0)
  V2  Regime exit  enter when regime=1, exit on flip/5d/reversal  [TEST K]
  V3  Short-only   only short signals, regime=1 filter            [TEST L]
  V4  Size-scaled  weight = clip((VIX/VIX3M - 0.9)/0.2, 0, 1)     [TEST M]

Per variant, output:
  - Sharpe / Ret / MDD / trades
  - Permutation p-value (shuffle signal timing, 500 iters)
  - Deflated Sharpe t-stat  (n_tests=30 across the whole project)
  - corr(pnl, SPY)
  - split-sample Sharpe
  - long vs short asymmetry
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings; warnings.filterwarnings('ignore')

# ============================================================
# DATA + FEATURES
# ============================================================
print("Downloading data...")
tickers = ["IEF", "^VIX", "^VIX3M", "SPY"]
raw = yf.download(tickers, start="2015-01-01", auto_adjust=False, progress=False)
df = raw['Close'].copy()
df.columns = [c.replace('^', '') for c in df.columns]
df = df.ffill().dropna()

ief_ret = df['IEF'].pct_change()
vol20   = ief_ret.rolling(20).std()
df['mom_20_z'] = df['IEF'].pct_change(20) / (vol20 * np.sqrt(20))
df['vix_ts']   = df['VIX'] / df['VIX3M']
df['regime']   = (df['vix_ts'] > 1.0).astype(int)
df['fwd5']     = df['IEF'].shift(-5) / df['IEF'] - 1
print(f"Range: {df.index[0].date()} -> {df.index[-1].date()}  n={len(df)}\n")

def make_signal(df, z=1.25):
    s = pd.Series(0.0, index=df.index)
    s[df['mom_20_z'] >  z] =  1.0
    s[df['mom_20_z'] < -z] = -1.0
    return s

def sharpe(r):
    return r.mean()*252 / (r.std()*np.sqrt(252)) if r.std()>0 else 0.0

def mdd(pnl):
    c = (1+pnl).cumprod()
    return (c/c.cummax() - 1).min()

# ============================================================
# SIMULATORS  (all take signal Series + df, no internal signal calc)
# ============================================================
COST = 1.0  # bps per side

def sim_baseline(df, sig, cost_bps=COST):
    pos = sig.replace(0, np.nan).ffill(limit=4).shift(1).fillna(0)
    daily = df['IEF'].pct_change().fillna(0)
    turn = pos.diff().abs().fillna(pos.abs())
    return pos, (pos * daily - turn * cost_bps/10000).fillna(0)

def sim_regime_binary(df, sig, cost_bps=COST):
    reg = df['regime'].shift(1).fillna(0)
    sig_f = sig.where(reg.eq(1), 0.0)
    return sim_baseline(df, sig_f, cost_bps)

def sim_regime_exit(df, sig, cost_bps=COST):
    """Enter when regime=1; exit on flip to 0, 5d cap, or reversal."""
    reg = df['regime'].shift(1).fillna(0).values
    sig_arr = sig.values
    pos_arr = np.zeros(len(df))
    direction = 0; days_held = 0
    for i in range(1, len(df)):
        s = sig_arr[i-1]
        r = reg[i-1]
        if direction == 0:
            if s != 0 and r == 1:
                direction = s; days_held = 1
        else:
            days_held += 1
            if r == 0 or days_held > 5 or (s != 0 and np.sign(s) == -np.sign(direction)):
                direction = 0; days_held = 0
        pos_arr[i] = direction
    pos = pd.Series(pos_arr, index=df.index)
    daily = df['IEF'].pct_change().fillna(0)
    turn = pos.diff().abs().fillna(pos.abs())
    return pos, (pos * daily - turn * cost_bps/10000).fillna(0)

def sim_short_only(df, sig, cost_bps=COST):
    reg = df['regime'].shift(1).fillna(0)
    sig_s = sig.where(sig < 0, 0.0).where(reg.eq(1), 0.0)
    return sim_baseline(df, sig_s, cost_bps)

def sim_size_scaled(df, sig, cost_bps=COST):
    """Weight = clip((VIX/VIX3M - 0.9)/0.2, 0, 1). Enter when signal fires."""
    vix_ts = df['vix_ts'].shift(1).fillna(0.9).values
    sig_arr = sig.values
    pos_arr = np.zeros(len(df))
    direction = 0; days_held = 0
    for i in range(1, len(df)):
        s = sig_arr[i-1]
        v = vix_ts[i-1]
        w = float(np.clip((v - 0.9)/0.2, 0.0, 1.0))
        if direction == 0:
            if s != 0 and w > 0:
                direction = s; days_held = 1
        else:
            days_held += 1
            if days_held > 5 or (s != 0 and np.sign(s) == -np.sign(direction)):
                direction = 0; days_held = 0
        pos_arr[i] = direction * w if direction != 0 else 0.0
    pos = pd.Series(pos_arr, index=df.index)
    daily = df['IEF'].pct_change().fillna(0)
    turn = pos.diff().abs().fillna(pos.abs())
    return pos, (pos * daily - turn * cost_bps/10000).fillna(0)

# ============================================================
# TEST HELPERS
# ============================================================
def perm_test(df, sig, sim_fn, n_iter=500, cost_bps=COST, seed=42):
    rng = np.random.default_rng(seed)
    sig_arr = sig.values.copy()
    sig_idx = np.where(sig_arr != 0)[0]
    if len(sig_idx) < 10:
        return np.nan, np.array([0.0])
    _, pnl_real = sim_fn(df, sig, cost_bps)
    real_sh = sharpe(pnl_real)
    null = np.empty(n_iter)
    for i in range(n_iter):
        shuf = np.zeros_like(sig_arr)
        shuf[rng.permutation(sig_idx)] = sig_arr[sig_idx]
        _, p2 = sim_fn(df, pd.Series(shuf, index=df.index), cost_bps)
        null[i] = sharpe(p2)
    return real_sh, null

def deflated_sharpe(pnl, n_tests=30):
    """Bailey & Lopez de Prado deflated SR t-stat."""
    p = pnl.dropna()
    if len(p) < 100: return np.nan
    T = len(p)
    skew = p.skew(); kurt = p.kurt()
    sr = sharpe(p)
    e_max = norm.ppf(1 - 1/n_tests) / np.sqrt(252)
    denom = np.sqrt(max(1 - skew*sr + (kurt-1)/4*sr*sr, 1e-9))
    return (sr - e_max) * np.sqrt(T/252) / denom

def split_sample(pnl):
    out = {}
    for lbl, s, e in [("pre-COVID","2015-01-01","2020-01-01"),
                      ("macros",   "2020-01-01","2023-01-01"),
                      ("recent",   "2023-01-01","2027-01-01")]:
        p = pnl.loc[s:e]
        out[lbl] = sharpe(p) if len(p)>30 else np.nan
    return out

def long_short_split(pnl, pos):
    ld = pnl[pos > 0.5]; sd = pnl[pos < -0.5]
    return (sharpe(ld) if len(ld)>20 else np.nan,
            sharpe(sd) if len(sd)>20 else np.nan)

def spy_corr(pnl):
    spy = df['SPY'].pct_change().reindex(pnl.index).fillna(0)
    return pnl.corr(spy)

# ============================================================
# RUN ALL VARIANTS
# ============================================================
sig = make_signal(df, z=1.25)

variants = [
    ("V0 Baseline",    sim_baseline),
    ("V1 Binary gate", sim_regime_binary),
    ("V2 Regime exit", sim_regime_exit),   # Test K
    ("V3 Short-only",  sim_short_only),    # Test L
    ("V4 Size-scaled", sim_size_scaled),   # Test M
]

results = {}
for name, fn in variants:
    pos, pnl = fn(df, sig)
    sh = sharpe(pnl)
    r  = pnl.sum()
    d  = mdd(pnl)
    n_tr = int((pos.diff().abs() > 0.05).sum())
    in_mkt = (pos.abs() > 0.05).mean()
    dsr = deflated_sharpe(pnl, n_tests=30)
    ss = split_sample(pnl)
    ls = long_short_split(pnl, pos)
    sc = spy_corr(pnl)
    # permutation (500 iters)
    print(f"  permuting {name}...", end=" ", flush=True)
    _, null = perm_test(df, sig, fn, n_iter=500)
    pval = (null >= sh).mean() if null.std() > 0 else np.nan
    print("done")
    results[name] = dict(pos=pos, pnl=pnl, sharpe=sh, ret=r, mdd=d, trades=n_tr,
                         in_mkt=in_mkt, dsr=dsr, ss=ss, ls=ls, spy_corr=sc, pval=pval)

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "="*110)
print("SUMMARY — ALL VARIANTS")
print("="*110)
print(f"{'Variant':16s} {'Sharpe':>7s} {'Ret':>8s} {'MDD':>8s} {'Trades':>7s} {'inMkt':>6s} "
      f"{'perm-p':>7s} {'DSR-t':>6s} {'SPYcor':>7s}")
print("-" * 110)
for name, r in results.items():
    print(f"{name:16s} {r['sharpe']:>+7.2f} {r['ret']*100:>+7.2f}% "
          f"{r['mdd']*100:>+7.1f}% {r['trades']:>7d} {r['in_mkt']*100:>5.0f}% "
          f"{r['pval']:>7.3f} {r['dsr']:>+6.2f} {r['spy_corr']:>+7.2f}")

# ============================================================
# DETAILED BREAKDOWN
# ============================================================
print("\n" + "="*110)
print("DETAIL PER VARIANT")
print("="*110)
for name, r in results.items():
    print(f"\n--- {name} ---")
    print(f"  Sharpe={r['sharpe']:+.2f}  Ret={r['ret']*100:+.2f}%  MDD={r['mdd']*100:+.1f}%  "
          f"Trades={r['trades']}  inMkt={r['in_mkt']*100:.0f}%")
    print(f"  Split sample:  pre-COVID={r['ss']['pre-COVID']:+.2f}  "
          f"macros={r['ss']['macros']:+.2f}  recent={r['ss']['recent']:+.2f}")
    print(f"  Long/Short:    Long={r['ls'][0]:+.2f}  Short={r['ls'][1]:+.2f}")
    print(f"  Perm p-value:  {r['pval']:.3f}  {'<0.05 ✓' if r['pval']<0.05 else '>0.05 ✗'}")
    print(f"  Deflated SR:   t={r['dsr']:+.2f}  {'>2 ✓' if r['dsr']>2 else '<2 ✗'}")
    print(f"  corr(SPY):     {r['spy_corr']:+.3f}")

# ============================================================
# BEST VARIANT — additional drilldown
# ============================================================
tradeable = [(n, r) for n, r in results.items()
             if r['pval'] < 0.05 and r['dsr'] > 1.5 and r['sharpe'] > 0.30]
if tradeable:
    best_name, best = max(tradeable, key=lambda x: x[1]['sharpe'])
else:
    best_name, best = max(results.items(), key=lambda x: x[1]['sharpe'])

print("\n" + "="*110)
print(f"BEST VARIANT: {best_name}  (Sharpe={best['sharpe']:+.2f})")
print("="*110)

pnl_best = best['pnl']

# Leave-one-year-out
print("\nLeave-one-year-out:")
base_sh = sharpe(pnl_best)
print(f"  Baseline Sharpe: {base_sh:+.2f}")
for y in sorted(pnl_best.index.year.unique()):
    if y == pnl_best.index.year.max(): continue
    p = pnl_best[pnl_best.index.year != y]
    sh = sharpe(p)
    flag = " *** CARRY" if (sh - base_sh) < -0.15 else ""
    print(f"  drop {y}: Sharpe={sh:+.2f}  Δ={sh-base_sh:+.2f}{flag}")

# Regime breakdown
print("\nRegime breakdown:")
for lbl, m in [
    ("VIX TS > 1.0", df['vix_ts'] > 1.0),
    ("VIX TS 0.9–1.0", (df['vix_ts']>=0.9)&(df['vix_ts']<=1.0)),
    ("VIX TS < 0.9", df['vix_ts'] < 0.9),
]:
    mask = m.reindex(pnl_best.index).fillna(False)
    p = pnl_best[mask]
    if len(p) > 30:
        print(f"  {lbl:15s}  n={len(p):4d}  Sharpe={sharpe(p):+.2f}  ret={p.sum()*100:+.2f}%")

# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*110)
print("VERDICT")
print("="*110)

for name, r in results.items():
    passes = []
    if r['sharpe'] > 0.30: passes.append("Sharpe>0.3")
    if r['pval'] < 0.05:   passes.append("perm-p<.05")
    if r['dsr'] > 2.0:     passes.append("DSR-t>2")
    if r['ss']['recent'] > 0: passes.append("recent-OOS>0")
    if abs(r['spy_corr']) < 0.3: passes.append("|SPYcor|<.3")
    n_pass = len(passes)
    if n_pass >= 4:
        verdict = "TRADEABLE (0.25-0.5x Kelly)"
    elif n_pass == 3:
        verdict = "PAPER-TRADE ONLY"
    else:
        verdict = "DO NOT TRADE"
    print(f"{name:16s}  {n_pass}/5 pass [{', '.join(passes)}]  -> {verdict}")

print("\nTests: Sharpe>0.3 | perm-p<0.05 | DSR-t>2 | recent>0 | |SPYcor|<0.3")
print("TRADEABLE if all 5 pass. PAPER-TRADE if 3-4. DO NOT TRADE if <3.")
print("="*110)
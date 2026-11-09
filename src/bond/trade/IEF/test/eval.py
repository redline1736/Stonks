import yfinance as yf
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')

# ==================== DATA ====================
print("Downloading...")
tickers = ["IEF","^TNX","^VIX","^VIX3M","^MOVE","^IRX","^FVX","^TYX","TLT","IEI","ZN=F"]
raw = yf.download(tickers, start="2015-01-01", auto_adjust=False, progress=False)
df = raw['Close'].copy()
df.columns = [c.replace('^','').replace('=F','') for c in df.columns]
df = df.ffill().dropna()

ief_ret   = df['IEF'].pct_change()
df['d20'] = df['IEF'].pct_change(20)
df['d5']  = df['IEF'].pct_change(5)
df['rv20']= ief_ret.rolling(20).std() * np.sqrt(252)

d = df['IEF'].diff()
g = d.clip(lower=0).rolling(14).mean()
l = (-d.clip(upper=0)).rolling(14).mean()
df['rsi'] = 100 - 100 / (1 + g / (l + 1e-9))

for h in [1, 3, 5, 10, 20]:
    df[f'fwd{h}'] = df['IEF'].shift(-h) / df['IEF'] - 1

base = df['fwd5'].gt(0).mean()
print(f"Base P(up 5d) = {base:.4f}   n={df['fwd5'].notna().sum()}\n")

buckets = {
    "20D < -3%": df['d20'] < -0.03,
    "20D > +3%": df['d20'] >  0.03,
    "RSI > 75":  df['rsi'] > 75,
}

# ==================== TEST 1: CLUSTERING ====================
def event_clusters(mask, min_gap=5):
    idx = np.where(mask.fillna(False).values)[0]
    if len(idx)==0: return pd.DataFrame(columns=['n_days'])
    brk = np.where(np.diff(idx) > min_gap)[0]
    st  = np.r_[idx[0], idx[brk+1]]
    en  = np.r_[idx[brk], idx[-1]]
    return pd.DataFrame({'n_days': en - st + 1})

print("="*72); print("TEST 1: EVENT CLUSTERING  (want events >= 60, mean_dur <= 3)")
print("="*72)
print(f"{'Bucket':12s} {'rows':>5s} {'events':>7s} {'mean_dur':>10s} {'max_dur':>8s} {'verdict':>10s}")
for name, m in buckets.items():
    cl = event_clusters(m)
    v = "OK" if (len(cl)>=60 and cl.n_days.mean()<=3) else "TOO CLUSTERED"
    print(f"{name:12s} {m.sum():>5d} {len(cl):>7d} {cl.n_days.mean():>10.1f} {cl.n_days.max():>8d} {v:>10s}")

# ==================== TEST 2: YEAR-BY-YEAR ====================
print("\n"+"="*72); print("TEST 2: YEAR-BY-YEAR  (want >= 5/9 years consistent)")
print("="*72)
df['year'] = df.index.year
df['y_up'] = np.where(df['fwd5'].notna(), (df['fwd5']>0).astype(float), np.nan)

for name, want_low in [("20D < -3%", True), ("20D > +3%", False)]:
    m = buckets[name]
    g = df.loc[m & df['y_up'].notna()].groupby('year')['y_up'].agg(['count','mean'])
    print(f"\n{name}  (want P(up) {'<0.45' if want_low else '>0.55'})")
    hits = 0; total = 0
    for yr, row in g.iterrows():
        if row['count'] < 5: continue
        total += 1
        ok = (row['mean']<0.45) if want_low else (row['mean']>0.55)
        if ok: hits += 1
        mark = " <-- hit" if ok else ""
        print(f"  {yr}: n={int(row['count']):3d}  P(up)={row['mean']:.3f}{mark}")
    print(f"  CONSISTENCY: {hits}/{total} years")

# ==================== TEST 3: HORIZON ROBUSTNESS ====================
print("\n"+"="*72); print("TEST 3: HORIZON ROBUSTNESS  (want edge across most horizons)")
print("="*72)
print(f"{'H':>4s} {'P(up|<-3%)':>11s} {'P(up|>+3%)':>11s} {'edge_dn':>8s} {'edge_up':>8s} {'n_dn':>5s} {'n_up':>5s}")
for h in [1, 3, 5, 10, 20]:
    fd = df[f'fwd{h}']
    p_dn = fd[df['d20'] < -0.03].gt(0).mean()
    p_up = fd[df['d20'] >  0.03].gt(0).mean()
    print(f"{h:>3d}d {p_dn:>11.3f} {p_up:>11.3f} {p_dn-0.5:>+8.3f} {p_up-0.5:>+8.3f} "
          f"{fd[df['d20']<-0.03].notna().sum():>5d} {fd[df['d20']>0.03].notna().sum():>5d}")

# ==================== TEST 4: THRESHOLD SENSITIVITY ====================
print("\n"+"="*72); print("TEST 4: THRESHOLD SENSITIVITY  (want monotonic, not a spike at 3%)")
print("="*72)
print(f"{'thr':>7s} {'n_dn':>5s} {'P(up|dn)':>10s} {'n_up':>5s} {'P(up|up)':>10s}")
for th in [0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050]:
    m_d = df['d20'] < -th; m_u = df['d20'] > th
    p_d = df.loc[m_d,'fwd5'].gt(0).mean() if m_d.sum() else np.nan
    p_u = df.loc[m_u,'fwd5'].gt(0).mean() if m_u.sum() else np.nan
    print(f"{th*100:>6.1f}% {m_d.sum():>5d} {p_d:>10.3f} {m_u.sum():>5d} {p_u:>10.3f}")

# ==================== TEST 5: BLOCK BOOTSTRAP CI ====================
print("\n"+"="*72); print("TEST 5: BLOCK BOOTSTRAP 95% CI  (need CI to exclude 0.50)")
print("="*72)
def block_bootstrap(vals, n=2000, block=5, seed=42):
    rng = np.random.default_rng(seed); L = len(vals); nb = L // block
    means = np.empty(n)
    for i in range(n):
        s = rng.integers(0, L-block, size=nb)
        idx = np.concatenate([np.arange(x, x+block) for x in s])
        means[i] = (vals[idx] > 0).mean()
    return means

for name, m in [("20D < -3%", buckets["20D < -3%"]), ("20D > +3%", buckets["20D > +3%"])]:
    f = df.loc[m,'fwd5'].dropna().values
    if len(f) < 30: continue
    boot = block_bootstrap(f)
    obs = (f > 0).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ok = (lo > 0.5) or (hi < 0.5)
    print(f"{name:10s} obs={obs:.3f}  CI=[{lo:.3f},{hi:.3f}]  "
          f"{'SIGNIFICANT' if ok else 'includes 0.50 — NOT significant'}")

# ==================== QUICK IC CHECK ====================
print("\n"+"="*72); print("IC: corr(20D past return, 5D forward return)")
print("="*72)
sub = df[['d20','fwd5']].dropna()
print(f"Pearson IC = {sub['d20'].corr(sub['fwd5']):.4f}")
print(f"Spearman IC = {sub['d20'].corr(sub['fwd5'], method='spearman'):.4f}")
print("(Positive IC = momentum. |IC| < 0.03 = essentially zero)")

# ==================== BACKTEST ====================
print("\n"+"="*72); print("BACKTEST: hold signal for 5 days, threshold + cost sweep")
print("="*72)

def backtest(df, th=0.025, hold=5, cost_bps=1.0):
    sig = pd.Series(0.0, index=df.index)
    sig[df['d20'] >  th] =  1.0
    sig[df['d20'] < -th] = -1.0

    # position today = average of signals from the last `hold` days
    pos = pd.Series(0.0, index=df.index)
    for lag in range(1, hold+1):
        pos = pos + sig.shift(lag).fillna(0) / hold

    daily = df['IEF'].pct_change().fillna(0)
    turn  = pos.diff().abs().fillna(pos.abs())
    ret   = pos * daily - turn * cost_bps/10000
    ret   = ret.fillna(0)

    ann_ret = ret.mean()*252
    ann_vol = ret.std()*np.sqrt(252)
    sh = ann_ret/ann_vol if ann_vol>0 else np.nan
    cum = (1+ret).cumprod()
    mdd = (cum/cum.cummax() - 1).min()
    return dict(ret=ann_ret, vol=ann_vol, sharpe=sh, mdd=mdd,
                turn=turn.sum()/len(turn)*252, in_mkt=(pos.abs()>0).mean(),
                n_yrs=len(df)/252)

print(f"{'thr':>6s} {'cost':>5s} {'ret':>9s} {'vol':>7s} {'Sharpe':>7s} {'MDD':>8s} {'turn/yr':>8s} {'in_mkt':>7s}")
for th in [0.02, 0.025, 0.03, 0.04]:
    for c in [0.5, 1.0, 3.0]:
        r = backtest(df, th=th, hold=5, cost_bps=c)
        print(f"{th*100:>5.1f}% {c:>4.1f}b {r['ret']*100:>+8.2f}% "
              f"{r['vol']*100:>6.2f}% {r['sharpe']:>7.2f} "
              f"{r['mdd']*100:>+7.2f}% {r['turn']:>8.1f} {r['in_mkt']*100:>6.0f}%")

bh = df['IEF'].pct_change().fillna(0)
bh_sh = (bh.mean()*252) / (bh.std()*np.sqrt(252))
bh_cm = (1+bh).cumprod(); bh_mdd = (bh_cm/bh_cm.cummax()-1).min()
print(f"\nBUY & HOLD: ret={bh.mean()*252*100:+.2f}%  Sharpe={bh_sh:.2f}  MDD={bh_mdd*100:.2f}%")
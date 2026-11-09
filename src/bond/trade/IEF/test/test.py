import yfinance as yf, pandas as pd, numpy as np

tickers = ["IEF","^TNX","^VIX","^VIX3M","^MOVE","^IRX","^FVX","^TYX","TLT","IEI","ZN=F"]
raw = yf.download(tickers, start="2015-01-01", auto_adjust=False, progress=False)
df  = raw['Close'].copy()
df.columns = [c.replace('^','').replace('=F','') for c in df.columns]
df = df.ffill().dropna()

ief_ret = df['IEF'].pct_change()
ief_5d  = df['IEF'].pct_change(5)
fwd_5d  = df['IEF'].shift(-5) / df['IEF'] - 1
base_up = fwd_5d.gt(0).mean()
print(f"Base P(up) = {base_up:.3f}   n = {fwd_5d.notna().sum()}\n")

# RSI(14)
d = df['IEF'].diff()
g = d.clip(lower=0).rolling(14).mean()
l = (-d.clip(upper=0)).rolling(14).mean()
rsi = 100 - 100 / (1 + g / (l + 1e-9))

# Vol regime
rv20 = ief_ret.rolling(20).std() * np.sqrt(252)
rv_pct = rv20.rolling(252).rank(pct=True)   # percentile of realized vol, last year

buckets = {
    "RSI < 25":              rsi < 25,
    "RSI 25–35":             (rsi >= 25) & (rsi < 35),
    "RSI 35–65 (neutral)":   (rsi >= 35) & (rsi < 65),
    "RSI 65–75":             (rsi >= 65) & (rsi < 75),
    "RSI > 75":              rsi > 75,
    "5D ret < -1.5%":        ief_5d < -0.015,
    "5D ret > +1.5%":        ief_5d >  0.015,
    "20D ret < -3%":         df['IEF'].pct_change(20) < -0.03,
    "20D ret > +3%":         df['IEF'].pct_change(20) >  0.03,
    "RV20 > 80th pctile":    rv_pct > 0.80,
    "RV20 < 20th pctile":    rv_pct < 0.20,
    "VIX term < 0.90 (calm)": df['VIX']/df['VIX3M'] < 0.90,
    "VIX term > 1.10 (stress)": df['VIX']/df['VIX3M'] > 1.10,
    "MOVE > 1.5σ trailing":  df['MOVE'] > df['MOVE'].rolling(60).mean()
                                         + 1.5*df['MOVE'].rolling(60).std(),
    "YC 10Y-3M < 0 (inverted)": (df['TNX'] - df['IRX']) < 0,
}

print(f"{'Bucket':30s} {'n':>5s} {'P(up|bucket)':>14s} {'Edge vs base':>14s} {'fwd mean':>10s}")
print("-" * 78)
for name, mask in buckets.items():
    mask = mask & fwd_5d.notna()
    n = mask.sum()
    if n < 30: 
        print(f"{name:30s} {n:5d}   (too few)")
        continue
    p_up = fwd_5d[mask].gt(0).mean()
    fwd_mean = fwd_5d[mask].mean()
    edge = p_up - base_up
    flag = "  <-- EDGE" if abs(edge) > 0.05 and n > 60 else ""
    print(f"{name:30s} {n:5d} {p_up:>14.3f} {edge:>+14.3f} {fwd_mean:>+10.4f}{flag}")
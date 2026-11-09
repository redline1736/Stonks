import yfinance as yf, pandas as pd, numpy as np, xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings; warnings.filterwarnings('ignore')

# ---------- 1. DATA ----------
tickers = ["SPY","^TNX","^VIX","^VIX3M","^MOVE","ZN=F","^FVX","^IRX",
           "IEF","TLT","^TYX","ZQ=F","HYG","LQD"]
raw = yf.download(tickers, start="2018-01-01", auto_adjust=False, progress=False)
df = raw['Close'].copy() if isinstance(raw.columns, pd.MultiIndex) else raw['Close'].to_frame()
df.columns = [c.replace('^','').replace('=F','') for c in df.columns]
df = df.ffill().dropna()

# ---------- 2. FEATURES ----------
f = pd.DataFrame(index=df.index)
ief_ret, spy_ret = df['IEF'].pct_change(), df['SPY'].pct_change()

# Returns
for t in ['SPY','IEF','TLT','HYG','LQD']:
    f[f'{t}_1M_ret'] = df[t].pct_change(21)
    f[f'{t}_5D_ret'] = df[t].pct_change(5)

# Rate changes (in bp — makes scales comparable)
f['TNX_5D_bp'] = (df['TNX'] - df['TNX'].shift(5)) * 100
f['TNX_1M_bp'] = (df['TNX'] - df['TNX'].shift(21)) * 100
f['TYX_5D_bp'] = (df['TYX'] - df['TYX'].shift(5)) * 100

# Yield curve
f['YC_10Y_3M']    = df['TNX'] - df['IRX']
f['YC_10Y_5Y']    = df['TNX'] - df['FVX']
f['YC_10Y_30T']   = df['TNX'] - df['TYX']
f['YC_change_5D'] = f['YC_10Y_3M'] - f['YC_10Y_3M'].shift(5)

# Fed funds futures
f['FF_rate']    = 100 - df['ZQ']
f['FF_5D_bp']   = (f['FF_rate'] - f['FF_rate'].shift(5)) * 100
f['FF_1M_bp']   = (f['FF_rate'] - f['FF_rate'].shift(21)) * 100
f['FF_TNX_sprd']= df['TNX'] - f['FF_rate']

# *** ZN basis: duration-adjusted (was the biggest broken feature) ***
# 10Y note futures duration ≈ 7.5.  A yield move of Δy (decimal) should move
# ZN price by ≈ -7.5·Δy·price.  Add the two return terms to get cheap/rich.
zn_ret5  = df['ZN'].pct_change(5)
tnx_chg5 = (df['TNX'] - df['TNX'].shift(5)) / 100.0
f['ZN_basis_5D'] = zn_ret5 + 7.5 * tnx_chg5

# Vol regime
f['VIX_TS']     = df['VIX'] / df['VIX3M']
f['VIX_1M_chg'] = df['VIX'] - df['VIX'].shift(21)
f['MOVE_VIX']   = df['MOVE'] / df['VIX']
f['MOVE_1M_chg']= df['MOVE'] - df['MOVE'].shift(21)

# Realized vol
f['IEF_RV20']      = ief_ret.rolling(20).std() * np.sqrt(252)
f['IEF_RV_ratio']  = f['IEF_RV20'] / f['IEF_RV20'].shift(21)

# Correlation regime
f['SPY_IEF_corr20'] = spy_ret.rolling(20).corr(ief_ret)
f['SPY_IEF_corr60'] = spy_ret.rolling(60).corr(ief_ret)
f['SPY_IEF_corr_d'] = f['SPY_IEF_corr20'] - f['SPY_IEF_corr60']

# Risk-on / risk-off
f['SPY_TLT_ratio']    = df['SPY'] / df['TLT']
f['SPY_TLT_ratio_1M'] = f['SPY_TLT_ratio'].pct_change(21)

# IEF RSI
delta = df['IEF'].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
f['IEF_RSI'] = 100 - 100 / (1 + gain / (loss + 1e-9))

# Credit stress (very informative for bond direction)
f['HYG_LQD_ratio']    = df['HYG'] / df['LQD']
f['HYG_LQD_ratio_1M'] = f['HYG_LQD_ratio'].pct_change(21)

f = f.replace([np.inf, -np.inf], np.nan)

# ---------- 3. TARGET ----------
# 5-day forward direction.  Simple direction (no dead-band) is more learnable
# than the 0.1% band — that band was throwing away ~30% of rows AND leaving
# only noise-level moves in the training set.
fwd = df['IEF'].shift(-5) / df['IEF'] - 1
target = (fwd > 0).astype(float).where(fwd.notna())

data = pd.concat([f, target.rename('y')], axis=1).dropna()
X, y = data.drop('y', axis=1), data['y'].astype(int)
print(f"Shape: {X.shape}  |  balance: {dict(y.value_counts())}")

# ---------- 4. PRE-SCREEN ----------
# Drop near-constant columns, then greedily drop >0.97 correlated pairs
X = X.loc[:, X.std() > 1e-8]
corr = X.corr().abs()
keep = []
for c in corr.columns:
    if all(corr.loc[c, k] < 0.97 for k in keep):
        keep.append(c)
X = X[keep]
print(f"After screen: {X.shape[1]} features kept")

# ---------- 5. WALK-FORWARD ----------
tscv = TimeSeriesSplit(n_splits=5)
model_accs, base_accs, aucs = [], [], []

for i, (tr, te) in enumerate(tscv.split(X)):
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=1.0,
        objective='binary:logistic', eval_metric='logloss',
        random_state=42, n_jobs=-1,
    )
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]

    # --- threshold: pick the one maximizing accuracy on the TRAIN fold,
    #     not a magic 0.60 ---
    ptr = m.predict_proba(Xtr)[:, 1]
    ths = np.linspace(0.35, 0.65, 61)
    acc_tr = [accuracy_score(ytr, (ptr > t).astype(int)) for t in ths]
    th = ths[int(np.argmax(acc_tr))]

    pred = (p > th).astype(int)
    base = int(ytr.mean() > 0.5)
    base_acc = accuracy_score(yte, np.full(len(yte), base))
    acc  = accuracy_score(yte, pred)
    auc  = roc_auc_score(yte, p) if yte.nunique() > 1 else np.nan

    model_accs.append(acc); base_accs.append(base_acc); aucs.append(auc)
    print(f"Fold {i+1} | th={th:.2f} | Model={acc:.4f} | Base={base_acc:.4f} | "
          f"AUC={auc:.3f} | p(mean)={p.mean():.3f} | #pred1={pred.sum()}/{len(pred)}")

print(f"\nAvg Model={np.mean(model_accs):.4f} | Avg Base={np.mean(base_accs):.4f} | "
      f"Avg AUC={np.nanmean(aucs):.3f}")
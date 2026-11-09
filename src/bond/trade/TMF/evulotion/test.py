import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ================================================================
# 1. DATA
# ================================================================
print("Downloading data...")
tickers = ["SPY", "^TNX", "^VIX", "^VIX3M", "^MOVE",
           "^IRX", "^FVX", "^TYX", "IEI", "IEF", "TLT", "TMF", "ZN=F", "ZQ=F"]

data_raw = yf.download(tickers, start="2020-01-01", auto_adjust=False, progress=False)
if data_raw.empty:
    raise ValueError("Data download failed.")

if isinstance(data_raw.columns, pd.MultiIndex):
    df = data_raw['Close'].copy()
else:
    df = data_raw['Close'].to_frame()

df.columns = [c.replace('^', '').replace('=F', '') for c in df.columns]
df = df.ffill().dropna()

# ================================================================
# 2. BASE FEATURES (your original set, de-duplicated)
# ================================================================
print("Building base features...")
features = pd.DataFrame(index=df.index)

features['SPY_1M_Return'] = df['SPY'].pct_change(21)
features['SPY_7M_Return'] = df['SPY'].pct_change(147)

features['IEF_1M_Return'] = df['IEF'].pct_change(21)
features['IEF_5D_Change'] = df['IEF'] - df['IEF'].shift(5)
features['IEF_1M_Change'] = df['IEF'] - df['IEF'].shift(21)

features['TNX_5D_Change'] = df['TNX'] - df['TNX'].shift(5)
features['TNX_1M_Change'] = df['TNX'] - df['TNX'].shift(21)
features['TNX_1M_Return'] = df['TNX'].pct_change(21)

features['TMF_5D_Change'] = df['TMF'].pct_change(5)
features['TMF_vs_TLT'] = features['TMF_5D_Change'] / df['TLT'].shift(5)

features['FF_Implied_Rate'] = 100 - df['ZQ']
features['FF_5D_Change'] = features['FF_Implied_Rate'] - features['FF_Implied_Rate'].shift(5)
features['FF_1M_Change'] = features['FF_Implied_Rate'] - features['FF_Implied_Rate'].shift(21)
features['FF_vs_TYX_Spread'] = df['TYX'] - features['FF_Implied_Rate']

features['ZQ_TMF_Inverse_Spread'] = df['ZQ'] - (100 - df['TMF'])
features['ZQ_TMF_Raw_Basis'] = df['ZQ'] - df['TMF']
features['ZQ_TLT_Inverse_Spread'] = df['ZQ'] - (100 - df['TLT'])
features['ZQ_TLT_Raw_Basis'] = df['ZQ'] - df['TLT']

features['TLT_1M_Return'] = df['TLT'].pct_change(21)
features['TLT_5D_Change'] = df['TLT'] - df['TLT'].shift(5)
features['TLT_TNX_Inverse_Spread'] = df['TLT'] - (100 - df['TYX'])
features['TLT_TYX_Inverse_Spread'] = df['TLT'] - (100 - df['TYX'])

features['TYX_5D_Change'] = df['TYX'] - df['TYX'].shift(5)
features['TYX_5D_Return'] = df['TYX'].pct_change(5)
features['TYX_1M_Change'] = df['TYX'] - df['TYX'].shift(21)

features['VIX_Term_Structure'] = df['VIX'] / df['VIX3M']
features['VIX_vs_1M_Ratio'] = df['VIX'] / df['VIX'].shift(21)
features['VIX_5D_Change'] = df['VIX'] - df['VIX'].shift(5)
features['VIX_1M_Change'] = df['VIX'] - df['VIX'].shift(21)
features['VIX_5D_Return'] = df['VIX'].pct_change(5)

features['MOVE_5D_Change'] = df['MOVE'] - df['MOVE'].shift(5)
features['MOVE_5D_Return'] = df['MOVE'].pct_change(5)
features['MOVE_1M_Change'] = df['MOVE'] - df['MOVE'].shift(21)
features['MOVE_VIX_Ratio'] = df['MOVE'] / df['VIX']

features['ZN_TYX_Inverse_Spread'] = df['ZN'] - (100 - df['TYX'])
features['ZN_TYX_Raw_Basis'] = df['ZN'] - df['TYX']
features['ZN_IEF_Raw_Basis'] = df['ZN'] - df['IEF']

features['Yield_Curve_30Y_3M'] = df['TYX'] - df['IRX']
features['Yield_Curve_30Y_5Y'] = df['TYX'] - df['FVX']
features['Yield_Curve_30Y_10Y'] = df['TYX'] - df['TNX']
features['Yield_Curve_Change_5D'] = (
    features['Yield_Curve_30Y_3M'] - features['Yield_Curve_30Y_3M'].shift(5)
)

features['Price_Curve_30Y_5Y'] = df['TMF'] - df['IEI']
features['Price_Curve_30Y_10Y'] = df['TMF'] - df['IEF']
features['Price_Curve_30Y_30T'] = df['TMF'] - df['TLT']
features['Price_Curve_30Y_5Y_Change_5D'] = (
    features['Price_Curve_30Y_5Y'] - features['Price_Curve_30Y_5Y'].shift(5)
)
features['Price_Curve_30Y_10Y_Change_5D'] = (
    features['Price_Curve_30Y_10Y'] - features['Price_Curve_30Y_10Y'].shift(5)
)
features['Price_Curve_30Y_30T_Change_5D'] = (
    features['Price_Curve_30Y_30T'] - features['Price_Curve_30Y_30T'].shift(5)
)

features['SPY_vs_TNX_1M_Preference'] = features['SPY_1M_Return'] - features['TNX_1M_Change'] / 100
features['SPY_TNX_Momentum_Divergence'] = features['SPY_1M_Return'] / (features['TNX_1M_Change'] + 0.001)

features['IEF_vs_TNX_1M_Preference'] = features['IEF_1M_Return'] - features['TNX_1M_Change'] / 100
features['IEF_vs_TNX_5D_Preference'] = features['IEF_5D_Change'] - features['TNX_5D_Change'] / 100

spy_ret = df['SPY'].pct_change()
tmf_ret = df['TMF'].pct_change()
features['SPY_TMF_Corr_60D'] = spy_ret.rolling(60).corr(tmf_ret)
features['SPY_TMF_Corr_20D'] = spy_ret.rolling(20).corr(tmf_ret)
features['SPY_TMF_Corr_Change'] = features['SPY_TMF_Corr_20D'] - features['SPY_TMF_Corr_60D']

features['SPY_vs_TYX_1M_Preference'] = features['SPY_1M_Return'] - features['TYX_1M_Change'] / 100
features['SPY_TYX_Momentum_Divergence'] = features['SPY_1M_Return'] / (features['TYX_1M_Change'] + 0.001)

features['TLT_vs_TYX_1M_Preference'] = features['TLT_1M_Return'] - features['TYX_1M_Change'] / 100
features['TLT_vs_TYX_5D_Preference'] = features['TLT_5D_Change'] - features['TYX_5D_Change'] / 100

n_base = features.shape[1]
print(f"Base features : {n_base}")

# ================================================================
# 3. 50 NEW HAND-CRAFTED FEATURES
# ================================================================
print("Building 50 new features...")

def rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def macd_hist(s, f=12, sl=26, sig=9):
    ef = s.ewm(span=f, adjust=False).mean()
    es = s.ewm(span=sl, adjust=False).mean()
    mac = ef - es
    return mac - mac.ewm(span=sig, adjust=False).mean()

def bb_pos(s, p=20):
    m = s.rolling(p).mean()
    sd = s.rolling(p).std()
    up = m + 2 * sd
    lo = m - 2 * sd
    return (s - lo) / (up - lo).replace(0, np.nan)

def range_pos(s, w=252):
    mn = s.rolling(w).min()
    mx = s.rolling(w).max()
    return (s - mn) / (mx - mn).replace(0, np.nan)

new = pd.DataFrame(index=df.index)

# --- Volatility (8) ---
new['N01_SPY_Vol_20']  = spy_ret.rolling(20).std()
new['N02_SPY_Vol_60']  = spy_ret.rolling(60).std()
new['N03_TLT_Vol_20']  = df['TLT'].pct_change().rolling(20).std()
new['N04_TMF_Vol_20']  = tmf_ret.rolling(20).std()
new['N05_TNX_Vol_20']  = df['TNX'].diff().rolling(20).std()
new['N06_VIX_Vol_20']  = df['VIX'].pct_change().rolling(20).std()
new['N07_VIX_Range_252'] = range_pos(df['VIX'])
new['N08_MOVE_Range_252'] = range_pos(df['MOVE'])

# --- Momentum (10) ---
new['N09_SPY_RSI_14'] = rsi(df['SPY'])
new['N10_TLT_RSI_14'] = rsi(df['TLT'])
new['N11_TMF_RSI_14'] = rsi(df['TMF'])
new['N12_TNX_RSI_14'] = rsi(df['TNX'])
new['N13_SPY_MACD']   = macd_hist(df['SPY'])
new['N14_TLT_MACD']   = macd_hist(df['TLT'])
new['N15_TMF_MACD']   = macd_hist(df['TMF'])
new['N16_SPY_SMA20_50'] = df['SPY'].rolling(20).mean() / df['SPY'].rolling(50).mean() - 1
new['N17_TLT_SMA20_50'] = df['TLT'].rolling(20).mean() / df['TLT'].rolling(50).mean() - 1
new['N18_TMF_SMA20_50'] = df['TMF'].rolling(20).mean() / df['TMF'].rolling(50).mean() - 1

# --- Distance from MAs (6) ---
new['N19_SPY_Dist_SMA20'] = df['SPY'] / df['SPY'].rolling(20).mean() - 1
new['N20_SPY_Dist_SMA50'] = df['SPY'] / df['SPY'].rolling(50).mean() - 1
new['N21_TLT_Dist_SMA20'] = df['TLT'] / df['TLT'].rolling(20).mean() - 1
new['N22_TLT_Dist_SMA50'] = df['TLT'] / df['TLT'].rolling(50).mean() - 1
new['N23_TMF_Dist_SMA20'] = df['TMF'] / df['TMF'].rolling(20).mean() - 1
new['N24_TMF_Dist_SMA50'] = df['TMF'] / df['TMF'].rolling(50).mean() - 1

# --- Bollinger (3) ---
new['N25_SPY_BB_Pos'] = bb_pos(df['SPY'])
new['N26_TLT_BB_Pos'] = bb_pos(df['TLT'])
new['N27_TMF_BB_Pos'] = bb_pos(df['TMF'])

# --- ATR-like (3) ---
new['N28_SPY_ATR_14'] = df['SPY'].diff().abs().rolling(14).mean() / df['SPY']
new['N29_TLT_ATR_14'] = df['TLT'].diff().abs().rolling(14).mean() / df['TLT']
new['N30_TMF_ATR_14'] = df['TMF'].diff().abs().rolling(14).mean() / df['TMF']

# --- Cross-asset correlations (8) ---
tlt_ret  = df['TLT'].pct_change()
tnx_diff = df['TNX'].diff()
vix_ret  = df['VIX'].pct_change()
move_ret = df['MOVE'].pct_change()
new['N31_SPY_TLT_Corr_20'] = spy_ret.rolling(20).corr(tlt_ret)
new['N32_SPY_TLT_Corr_60'] = spy_ret.rolling(60).corr(tlt_ret)
new['N33_SPY_TNX_Corr_20'] = spy_ret.rolling(20).corr(tnx_diff)
new['N34_TLT_TNX_Corr_20'] = tlt_ret.rolling(20).corr(tnx_diff)
new['N35_TMF_TNX_Corr_20'] = tmf_ret.rolling(20).corr(tnx_diff)
new['N36_SPY_VIX_Corr_20'] = spy_ret.rolling(20).corr(vix_ret)
new['N37_TLT_VIX_Corr_20'] = tlt_ret.rolling(20).corr(vix_ret)
new['N38_TMF_MOVE_Corr_20'] = tmf_ret.rolling(20).corr(move_ret)

# --- Yield curve extras (5) ---
new['N39_Yield_Curvature']        = 2 * df['TNX'] - df['FVX'] - df['TYX']
new['N40_Yield_Curvature_Chg_5D'] = new['N39_Yield_Curvature'] - new['N39_Yield_Curvature'].shift(5)
new['N41_TNX_IRX_Spread_Chg_5D']  = (df['TNX'] - df['IRX']) - (df['TNX'] - df['IRX']).shift(5)
new['N42_Bond_Risk_Premium']      = df['TYX'] - (100 - df['ZQ'])
new['N43_VIX_TNX_Ratio']          = df['VIX'] / df['TNX']

# --- Higher moments / drawdowns (4) ---
new['N44_SPY_Skew_20'] = spy_ret.rolling(20).skew()
new['N45_TLT_Skew_20'] = tlt_ret.rolling(20).skew()
new['N46_TMF_DD_60']   = df['TMF'] / df['TMF'].rolling(60).max() - 1
new['N47_TLT_DD_60']   = df['TLT'] / df['TLT'].rolling(60).max() - 1

# --- Misc (3) ---
new['N48_VIX_MOVE_Spread']    = df['VIX'] - df['MOVE']
new['N49_TMF_TLT_Ratio']      = df['TMF'] / df['TLT']
new['N50_TMF_TLT_Ratio_Chg_5D'] = new['N49_TMF_TLT_Ratio'] - new['N49_TMF_TLT_Ratio'].shift(5)

features = pd.concat([features, new], axis=1)
assert new.shape[1] == 50, f"Expected 50 new features, got {new.shape[1]}"
print(f"Total features: {features.shape[1]} ({n_base} base + 50 new)")

# ================================================================
# 4. TARGET + CLEAN
# ================================================================
future_tmf = df['TMF'].shift(-5)
df['Target'] = np.where(future_tmf > df['TMF'] * 1.01, 1,
               np.where(future_tmf < df['TMF'] * 0.99, 0, np.nan))

model_data = pd.concat([features, df['Target']], axis=1).dropna()
X = model_data.drop('Target', axis=1)
y = model_data['Target'].astype(int)

print(f"\nDataset shape  : {model_data.shape}")
print(f"Features       : {X.shape[1]}")
print(f"Class balance  : {y.value_counts().to_dict()}")

# ================================================================
# 5. GENETIC ALGORITHM
# ================================================================
print("\n" + "=" * 60)
print("GENETIC ALGORITHM — feature subset evolution")
print("=" * 60)

# Pre-compute CV splits on the aligned dataset
N_SPLITS = 5
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
splits = list(tscv.split(X))

XGB_PARAMS = dict(
    n_estimators=60, max_depth=3, learning_rate=0.06,
    subsample=0.8, colsample_bytree=0.8,
    objective='binary:logistic', eval_metric='logloss',
    random_state=42, n_jobs=-1, tree_method='hist',
)

def fitness(mask: np.ndarray) -> float:
    """Walk-forward accuracy for a given feature mask."""
    cols = X.columns[mask.astype(bool)]
    if len(cols) < 3:
        return 0.0
    Xs = X[cols]
    accs = []
    for tr, te in splits:
        try:
            m = xgb.XGBClassifier(**XGB_PARAMS)
            m.fit(Xs.iloc[tr], y.iloc[tr])
            p = (m.predict_proba(Xs.iloc[te])[:, 1] > 0.5).astype(int)
            accs.append(accuracy_score(y.iloc[te], p))
        except Exception:
            accs.append(0.5)
    return float(np.mean(accs))

def random_individual(n: int, min_feat: int = 5) -> np.ndarray:
    while True:
        ind = np.random.randint(0, 2, n).astype(bool)
        if ind.sum() >= min_feat:
            return ind

N_FEAT   = X.shape[1]
POP      = 20
GENS     = 12
ELITE    = 2
MIN_FEAT = 5
MUT_RATE = max(2.0 / N_FEAT, 0.02)

print(f"Population      : {POP}")
print(f"Generations     : {GENS}")
print(f"Feature space   : {N_FEAT}")
print(f"CV folds        : {N_SPLITS}")
print(f"Total model fits: ~{POP * GENS * N_SPLITS}\n")

pop = [random_individual(N_FEAT, MIN_FEAT) for _ in range(POP)]
best_ever, best_ever_fit = None, -1.0
history = []

for gen in range(GENS):
    fits = np.array([fitness(ind) for ind in pop])

    # Sort by fitness
    order = np.argsort(fits)[::-1]
    pop = [pop[i] for i in order]
    fits = fits[order]

    if fits[0] > best_ever_fit:
        best_ever_fit = fits[0]
        best_ever = pop[0].copy()

    history.append(dict(gen=gen + 1, best=float(fits[0]),
                        avg=float(fits.mean()), n=int(pop[0].sum())))

    print(f"Gen {gen+1:2d} | best={fits[0]:.4f} | avg={fits.mean():.4f} | "
          f"n_feat(best)={int(pop[0].sum()):3d} | ever={best_ever_fit:.4f}")

    # ----- build next generation -----
    new_pop = [p.copy() for p in pop[:ELITE]]           # elitism

    def tournament():
        idx = np.random.choice(POP, 3, replace=False)
        return pop[max(idx, key=lambda i: fits[i])]

    while len(new_pop) < POP:
        p1, p2 = tournament(), tournament()
        mask = np.random.rand(N_FEAT) < 0.5
        child = np.where(mask, p1, p2)
        mut = np.random.rand(N_FEAT) < MUT_RATE
        child = np.logical_xor(child, mut)
        if child.sum() < MIN_FEAT:
            child = random_individual(N_FEAT, MIN_FEAT)
        new_pop.append(child)

    pop = new_pop

# ================================================================
# 6. FINAL MODEL ON THE EVOLVED FEATURE SET
# ================================================================
best_cols = X.columns[best_ever]
print("\n" + "=" * 60)
print(f"WINNING FEATURES ({len(best_cols)}):")
print("=" * 60)
for c in best_cols:
    src = "ORIG" if c in features.columns[:n_base] else "NEW "
    print(f"  [{src}] {c}")

X_best = X[best_cols]

print("\n" + "=" * 60)
print("WALK-FORWARD EVALUATION ON EVOLVED SET")
print("=" * 60)

fold_acc, base_acc, fold_f1 = [], [], []
last_model = None
for i, (tr, te) in enumerate(splits):
    m = xgb.XGBClassifier(**XGB_PARAMS)
    m.fit(X_best.iloc[tr], y.iloc[tr])
    proba = m.predict_proba(X_best.iloc[te])[:, 1]
    pred = (proba > 0.55).astype(int)     # mild abstain threshold

    fold_acc.append(accuracy_score(y.iloc[te], pred))
    fold_f1.append(f1_score(y.iloc[te], pred, zero_division=0))
    maj = int(y.iloc[tr].mean() > 0.5)
    base_acc.append(accuracy_score(y.iloc[te], np.full(len(te), maj)))
    last_model = m

    print(f"Fold {i+1} | N={len(te):4d} | Acc={fold_acc[-1]:.4f} | "
          f"F1={fold_f1[-1]:.4f} | Base={base_acc[-1]:.4f}")

print(f"\nAverage evolved accuracy : {np.mean(fold_acc):.4f}")
print(f"Average baseline accuracy: {np.mean(base_acc):.4f}")
print(f"Average F1               : {np.mean(fold_f1):.4f}")
print(f"Beat baseline by         : {np.mean(fold_acc) - np.mean(base_acc):+.4f}")

print("\nFeature importances (last fold):")
imp = last_model.feature_importances_
for j in np.argsort(imp)[::-1]:
    print(f"  {best_cols[j]:32s}: {imp[j]:.4f}")
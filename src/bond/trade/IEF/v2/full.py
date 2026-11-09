import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 1. DOWNLOAD DATA
# ------------------------------------------------------------------
print("Downloading data...")
tickers = ["SPY", 
           "^TNX", 
           "^VIX", 
           "^VIX3M", 
           "^MOVE",
           "^IRX", # Yield 13 week bonds
           "^FVX", # Yield  5 year bonds
           "^TNX", # Yield 10 year bonds
           "^TYX", #  Yield  30 year bonds
           "IEI", # 5 year bonds 
           "IEF",  # 10 year bonds
           "TLT", # 20 year bonds
           "TMF", # 20 year bonds leverage
           "ZN=F", # Future ten year bonds
           "ZQ=F", # future 25 year bonds
           ]

data_raw = yf.download(tickers, start="2020-01-01", auto_adjust=False, progress=False)
if data_raw.empty:
    raise ValueError("Data download failed.")

# Extract Close prices (handles both single- and multi-level column frames)
if isinstance(data_raw.columns, pd.MultiIndex):
    df = data_raw['Close'].copy()
else:
    df = data_raw['Close'].to_frame()

df.columns = [c.replace('^', '').replace('=F', '') for c in df.columns]
df = df.ffill().dropna()

# ------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ------------------------------------------------------------------
print("Building features...")
features = pd.DataFrame(index=df.index)

# SPY
features['SPY_1M_Return'] = df['SPY'].pct_change(21)
features['SPY_7M_Return'] = df['SPY'].pct_change(147)

# IEF
features['IEF_1M_Return'] = df['IEF'].pct_change(21)
features['IEF_5D_Change'] = df['IEF']  - df['IEF'].shift(5)
features['IEF_1M_Change'] = df['IEF'] - df['IEF'].shift(21)

ief_ret = df['IEF'].pct_change()
features['IEF_RealVol_20D'] = ief_ret.rolling(20).std() * np.sqrt(252)
features['IEF_RealVol_Change'] = (
    features['IEF_RealVol_20D'] / features['IEF_RealVol_20D'].shift(21) - 1
)

# TNX
features['TNX_5D_Change'] = df['TNX'] - df['TNX'].shift(5)
features['TNX_1M_Change'] = df['TNX'] - df['TNX'].shift(21)
features['TNX_1M_Return'] = df['TNX'].pct_change(21)

# TMF
features['TMF_5D_Change'] = df['TMF'].pct_change(5)
features['TMF_vs_TLT'] = features['TMF_5D_Change'] / df['TLT'].shift(5)
features['TMF_TNX_Inverse_Spread'] = df['TMF'] - (100 - df['TNX'])

# ZQ
features['FF_Implied_Rate'] = 100 - df['ZQ']               # implied FF rate
features['FF_5D_Change']    = features['FF_Implied_Rate'] - features['FF_Implied_Rate'].shift(5)
features['FF_1M_Change']    = features['FF_Implied_Rate'] - features['FF_Implied_Rate'].shift(21)
features['FF_vs_TNX_Spread'] = df['TNX'] - features['FF_Implied_Rate']

# TLT 
features['TLT_1M_Return'] = df['TLT'].pct_change(21)
features['TLT_5d_Change'] = df['TLT']  - df['TLT'].shift(5)

features['TLT_TNX_Inverse_Spread'] = df['TLT'] - (100 - df['TNX'])
# TYX
features['TYX_5d_Change'] = df['TYX'] -  df['TYX'].shift(5)

# SPY vs TNX
features['SPY_vs_TNX_1M_Preference'] = features['SPY_1M_Return'] - features['TNX_1M_Change'] / 100
features['SPY_TNX_Momentum_Divergence'] = features['SPY_1M_Return'] / (features['TNX_1M_Change'] + 0.001)


# IEF vs TNX
features['IEF_vs_TNX_1M_Preference'] = (
    features['IEF_1M_Return'] - features['TNX_1M_Change'] / 100
)
features['IEF_vs_TNX_5D_Preference'] = (
    features['IEF_5D_Change'] - features['TNX_5D_Change'] / 100
)

# SPY vs IEF
spy_ret = df['SPY'].pct_change()
ief_ret = df['IEF'].pct_change()
features['SPY_IEF_Corr_60D']  = spy_ret.rolling(60).corr(ief_ret)
features['SPY_IEF_Corr_20D']  = spy_ret.rolling(20).corr(ief_ret)
features['SPY_IEF_Corr_Change'] = (
    features['SPY_IEF_Corr_20D'] - features['SPY_IEF_Corr_60D']
)

# VIX
features['VIX_Term_Structure'] = df['VIX'] / df['VIX3M']
features['VIX_vs_1M_Ratio'] = df['VIX'] / df['VIX'].shift(21)

# MOVE
features['MOVE_5D_Change'] = df['MOVE'] - df['MOVE'].shift(5)
features['MOVE_1M_Change'] = df['MOVE'] - df['MOVE'].shift(21)
features['MOVE_VIX_Ratio'] = df['MOVE'] / df['VIX']

# ZN Futures
features['ZN_TNX_Inverse_Spread'] = df['ZN'] - (100 - df['TNX'])
features['ZN_TNX_Raw_Basis'] = df['ZN'] - df['TNX']
features['ZN_IEF_Raw_Basis'] = df['ZN'] - df['IEF']

# Yield curve
features['Yield_Curve_10Y_3M'] = df['TNX'] - df['IRX']
features['Yield_Curve_10Y_5Y'] = df['TNX'] - df['FVX']
features['Yield_Curve_10Y_30T'] = df['TNX'] - df['TYX']

features['Yield_Curve_Change_5D'] = (
    features['Yield_Curve_10Y_3M'] - features['Yield_Curve_10Y_3M'].shift(5)
)

# Price curve
features['Price_Curve_10Y_5Y'] = df['IEF'] - df['IEI']
features['Price_Curve_10Y_30T'] = df['TNX'] - df['TLT']
features['Price_Curve_10Y_30T_leverage'] = df['TNX'] - df['TMF']

features['Price_Curve_Change_5D'] = (
    features['Price_Curve_10Y_5Y'] - features['Price_Curve_10Y_5Y'].shift(5)
)





# ------------------------------------------------------------------
# 3. TARGET: IEF moves > 0.1% over next 5 trading days
# ------------------------------------------------------------------
future_ief = df['IEF'].shift(-5)
df['Target'] = np.where(
    future_ief > df['IEF'] * 1.001, 1,
    np.where(future_ief < df['IEF'] * 0.999, 0, np.nan)
)

# ------------------------------------------------------------------
# 4. CLEAN AND ALIGN
# ------------------------------------------------------------------
model_data = pd.concat([features, df['Target']], axis=1).dropna()
X = model_data.drop('Target', axis=1)
y = model_data['Target'].astype(int)     # dropna leaves {0.0, 1.0} → cast to int

print(f"\nDataset shape : {model_data.shape}")
print(f"Features       : {len(X.columns)}")
print(f"Class balance  : {y.value_counts().to_dict()}")

# ------------------------------------------------------------------
# 5. WALK-FORWARD VALIDATION
# ------------------------------------------------------------------
print("\nTraining XGBoost with Walk-Forward Validation...")
tscv = TimeSeriesSplit(n_splits=6)

fold_accuracies, baseline_accuracies = [], []
last = {}

for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    predictions = (probs > 0.60).astype(int)      # abstain from weak signals

    acc = accuracy_score(y_test, predictions)
    fold_accuracies.append(acc)

    # Baseline = majority class from *training* fold
    majority = int(y_train.mean() > 0.5)
    base_acc = accuracy_score(y_test, np.full(len(y_test), majority))
    baseline_accuracies.append(base_acc)

    print(f"Fold {i+1} | N_test={len(y_test):4d} | "
          f"Model={acc:.4f} | Baseline={base_acc:.4f}")

    last = dict(model=model, y_test=y_test, preds=predictions)

print("\n--- OVERALL RESULTS ---")
print(f"Average Model Accuracy : {np.mean(fold_accuracies):.4f}")
print(f"Average Baseline Acc   : {np.mean(baseline_accuracies):.4f}")

print("\nClassification Report (Last Fold):")
print(classification_report(last['y_test'], last['preds'], zero_division=0))

print("\nFeature Importances (Last Fold):")
importances = last['model'].feature_importances_
for j in np.argsort(importances)[::-1]:
    print(f"{X.columns[j]:30s}: {importances[j]:.4f}")
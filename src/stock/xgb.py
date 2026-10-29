import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost model
model = xgb.XGBClassifier(
    n_estimators=100,      # number of boosting rounds (trees)
    max_depth=3,           # max depth per tree
    learning_rate=0.1,     # shrinkage (eta)
    subsample=0.8,         # row sampling per tree
    colsample_bytree=0.8,  # feature sampling per tree
    reg_lambda=1.0,        # L2 regularization on leaf weights
    reg_alpha=0.0,         # L1 regularization on leaf weights
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print(f"AUC:      {roc_auc_score(y_test, probs):.4f}")
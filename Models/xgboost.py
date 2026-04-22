import json
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, make_scorer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from Dataset.preprocessing import (
    fiverr_dss_pipeline,
    X,
    y,
    X_train,
    X_test,
    y_train,
    y_test,
    text_col,
    cat_col,
    target_enc_col,
    num_cols,
)

# ──────────────────────────────────────────────
# 1. Transform features
# ──────────────────────────────────────────────
X_train_proc = fiverr_dss_pipeline.transform(X_train).toarray()
X_test_proc = fiverr_dss_pipeline.transform(X_test).toarray()

print("Transformed train shape:", X_train_proc.shape)
print("Transformed test shape:", X_test_proc.shape)

def dollar_r2(y_true, y_pred):
    return r2_score(np.expm1(y_true), np.expm1(y_pred))

def dollar_mae(y_true, y_pred):
    return mean_absolute_error(np.expm1(y_true), np.expm1(y_pred))

def dollar_mape(y_true, y_pred):
    return mean_absolute_percentage_error(np.expm1(y_true), np.expm1(y_pred))

def dollar_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred)))

scoring = {
    "r2":   make_scorer(dollar_r2),
    "mae":  make_scorer(dollar_mae, greater_is_better=False),
    "mape": make_scorer(dollar_mape, greater_is_better=False),
    "rmse": make_scorer(dollar_rmse, greater_is_better=False),
}

# ──────────────────────────────────────────────
# 2. Load Optuna-tuned params & build model
# ──────────────────────────────────────────────
params_path = Path(__file__).resolve().parent.parent / "xgb_best_params.json"
with open(params_path) as f:
    best_params = json.load(f)

best_params.update({"random_state": 42, "verbosity": 0, "n_jobs": -1})
xgb = XGBRegressor(**best_params)

# ──────────────────────────────────────────────
# 3. K-Fold Cross-Validation (dollar-space)
# ──────────────────────────────────────────────
full_pipeline = Pipeline(
    steps=[
        ("preprocess", fiverr_dss_pipeline),
        ("model", xgb),
    ]
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv = cross_validate(full_pipeline, X, y, cv=kf, scoring=scoring)

r2_scores  = cv["test_r2"]
mae_scores = -cv["test_mae"]
mape_scores = -cv["test_mape"]
rmse_scores = -cv["test_rmse"]

print("\n=== 5-Fold Cross-Validation (dollar-space) ===")
print(f"R²   : {np.round(r2_scores, 4)}  → Mean: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
print(f"MAE  : {np.round(mae_scores, 2)}  → Mean: ${mae_scores.mean():.2f} ± {mae_scores.std():.2f}")
print(f"MAPE : {np.round(mape_scores, 4)}  → Mean: {mape_scores.mean():.4f} ± {mape_scores.std():.4f}")
print(f"RMSE : {np.round(rmse_scores, 2)}  → Mean: ${rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")

# ──────────────────────────────────────────────
# 4. Train on full data for feature importance
# ──────────────────────────────────────────────
xgb.fit(X_train_proc, y_train)

# ──────────────────────────────────────────────
# 5. Feature Importance (top 10)
# ──────────────────────────────────────────────
enc = fiverr_dss_pipeline.named_steps["encoding"]
tfidf = enc.named_transformers_["tfidf"]
ohe = enc.named_transformers_["ohe"]

tfidf_features = tfidf.get_feature_names_out()
ohe_features = ohe.get_feature_names_out(cat_col)
target_features = np.array([f"{col}_te" for col in target_enc_col])
num_features = np.array(num_cols)

feature_names = np.concatenate(
    [tfidf_features, ohe_features, target_features, num_features]
)

feature_importance = xgb.feature_importances_
top_indices = np.argsort(feature_importance)[-10:][::-1]
print("\n=== Top 10 Most Important Features ===")
for idx in top_indices:
    name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
    print(f"  {name}: {feature_importance[idx]:.4f}")

import json
import numpy as np
import optuna
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, make_scorer

from Dataset.preprocessing import (
    fiverr_dss_pipeline,
    X,
    y,
    X_train,
    X_test,
    y_train,
    y_test,
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


# ──────────────────────────────────────────────
# Custom scorer: R² evaluated in dollar-space
# ──────────────────────────────────────────────
def dollar_r2(y_true, y_pred):
    return r2_score(np.expm1(y_true), np.expm1(y_pred))

dollar_r2_scorer = make_scorer(dollar_r2)


# ──────────────────────────────────────────────
# Helper: feature names from the pipeline
# ──────────────────────────────────────────────
def get_feature_names():
    enc = fiverr_dss_pipeline.named_steps["encoding"]
    tfidf = enc.named_transformers_["tfidf"]
    ohe = enc.named_transformers_["ohe"]
    tfidf_features = tfidf.get_feature_names_out()
    ohe_features = ohe.get_feature_names_out(cat_col)
    target_features = np.array([f"{col}_te" for col in target_enc_col])
    num_features_arr = np.array(num_cols)
    return np.concatenate([tfidf_features, ohe_features, target_features, num_features_arr])


def print_top_features(importances, n=10):
    feature_names = get_feature_names()
    top_indices = np.argsort(importances)[-n:][::-1]
    for idx in top_indices:
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        print(f"  {name}: {importances[idx]:.4f}")


def evaluate_dollar_space(y_true, y_pred, model_name):
    y_true_d = np.expm1(y_true)
    y_pred_d = np.expm1(y_pred)
    r2 = r2_score(y_true_d, y_pred_d)
    mae = mean_absolute_error(y_true_d, y_pred_d)
    mape = mean_absolute_percentage_error(y_true_d, y_pred_d)
    print(f"\n=== {model_name} Test Performance (dollar-space) ===")
    print(f"R²   : {r2:.4f}")
    print(f"MAE  : ${mae:.2f}")
    print(f"MAPE : {mape:.4f}")


# ══════════════════════════════════════════════
#  XGBOOST OPTIMISATION
# ══════════════════════════════════════════════

# ──────────────────────────────────────────────
# 2. XGBoost Optuna Objective
# ──────────────────────────────────────────────
def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
    }

    model = XGBRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_proc, y_train, cv=kf,
                             scoring=dollar_r2_scorer, n_jobs=-1)
    return scores.mean()


# ──────────────────────────────────────────────
# 3. Run XGBoost Optimization
# ──────────────────────────────────────────────
xgb_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    study_name="xgboost",
)

xgb_study.optimize(xgb_objective, n_trials=200, show_progress_bar=True)

print("\n=== XGBoost Optuna Results ===")
print(f"Best CV R² (dollar-space): {xgb_study.best_value:.4f}")
print(f"Best params: {xgb_study.best_params}")


# ──────────────────────────────────────────────
# 4. Train Final XGBoost with Best Parameters
# ──────────────────────────────────────────────
xgb_best = xgb_study.best_params.copy()
xgb_best.update({"random_state": 42, "verbosity": 0, "n_jobs": -1,
                  "early_stopping_rounds": 20})

best_xgb = XGBRegressor(**xgb_best)
best_xgb.fit(X_train_proc, y_train,
             eval_set=[(X_test_proc, y_test)], verbose=False)

# ──────────────────────────────────────────────
# 5. Evaluate XGBoost (dollar-space)
# ──────────────────────────────────────────────
evaluate_dollar_space(y_test, best_xgb.predict(X_test_proc), "Tuned XGBoost")

# ──────────────────────────────────────────────
# 6. XGBoost Full-Data CV (sanity check)
# ──────────────────────────────────────────────
xgb_cv = xgb_study.best_params.copy()
xgb_cv.update({"random_state": 42, "verbosity": 0, "n_jobs": -1})

xgb_pipeline = Pipeline([
    ("preprocess", fiverr_dss_pipeline),
    ("model", XGBRegressor(**xgb_cv)),
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_cv_scores = cross_val_score(xgb_pipeline, X, y, cv=kf,
                                 scoring=dollar_r2_scorer)

print("\n=== XGBoost 5-Fold CV on Full Data (dollar-space) ===")
print("Fold R² scores:", np.round(xgb_cv_scores, 4))
print(f"Mean R²: {xgb_cv_scores.mean():.4f}")
print(f"Std  R²: {xgb_cv_scores.std():.4f}")

# ──────────────────────────────────────────────
# 7. XGBoost Feature Importance
# ──────────────────────────────────────────────
print("\n=== XGBoost Top 10 Most Important Features ===")
print_top_features(best_xgb.feature_importances_)

# ──────────────────────────────────────────────
# 8. Save best params to JSON
# ──────────────────────────────────────────────
params_dir = Path(__file__).resolve().parent.parent
xgb_params_path = params_dir / "xgb_best_params.json"

with open(xgb_params_path, "w") as f:
    json.dump(xgb_study.best_params, f, indent=2)

print(f"\nSaved XGBoost best params → {xgb_params_path}")

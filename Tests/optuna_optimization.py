import json
import numpy as np
import optuna
import joblib
from pathlib import Path
from lightgbm import LGBMRegressor
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
#  LIGHTGBM OPTIMISATION
# ══════════════════════════════════════════════

# ──────────────────────────────────────────────
# 2a. LightGBM Optuna Objective
# ──────────────────────────────────────────────
def lgbm_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    max_leaves = 2 ** params["max_depth"]
    if params["num_leaves"] > max_leaves:
        params["num_leaves"] = max_leaves

    model = LGBMRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_proc, y_train, cv=kf,
                             scoring=dollar_r2_scorer, n_jobs=-1)
    return scores.mean()


# ──────────────────────────────────────────────
# 3a. Run LightGBM Optimization
# ──────────────────────────────────────────────
lgbm_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    study_name="lgbm",
)

lgbm_study.optimize(lgbm_objective, n_trials=200, show_progress_bar=True)

print("\n=== LightGBM Optuna Results ===")
print(f"Best CV R² (dollar-space): {lgbm_study.best_value:.4f}")
print(f"Best params: {lgbm_study.best_params}")


# ──────────────────────────────────────────────
# 4a. Train Final LightGBM with Best Parameters
# ──────────────────────────────────────────────
lgbm_best = lgbm_study.best_params.copy()
max_leaves = 2 ** lgbm_best["max_depth"]
if lgbm_best["num_leaves"] > max_leaves:
    lgbm_best["num_leaves"] = max_leaves
lgbm_best.update({"random_state": 42, "verbose": -1, "n_jobs": -1,
                   "early_stopping_rounds": 20})

best_lgbm = LGBMRegressor(**lgbm_best)
best_lgbm.fit(X_train_proc, y_train,
              eval_set=[(X_test_proc, y_test)], eval_metric="rmse")

# ──────────────────────────────────────────────
# 5a. Evaluate LightGBM (dollar-space)
# ──────────────────────────────────────────────
evaluate_dollar_space(y_test, best_lgbm.predict(X_test_proc), "Tuned LightGBM")

# ──────────────────────────────────────────────
# 6a. LightGBM Full-Data CV (sanity check)
# ──────────────────────────────────────────────
lgbm_cv = lgbm_study.best_params.copy()
max_leaves = 2 ** lgbm_cv["max_depth"]
if lgbm_cv["num_leaves"] > max_leaves:
    lgbm_cv["num_leaves"] = max_leaves
lgbm_cv.update({"random_state": 42, "verbose": -1, "n_jobs": -1})

lgbm_pipeline = Pipeline([
    ("preprocess", fiverr_dss_pipeline),
    ("model", LGBMRegressor(**lgbm_cv)),
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgbm_cv_scores = cross_val_score(lgbm_pipeline, X, y, cv=kf,
                                  scoring=dollar_r2_scorer)

print("\n=== LightGBM 5-Fold CV on Full Data (dollar-space) ===")
print("Fold R² scores:", np.round(lgbm_cv_scores, 4))
print(f"Mean R²: {lgbm_cv_scores.mean():.4f}")
print(f"Std  R²: {lgbm_cv_scores.std():.4f}")

# ──────────────────────────────────────────────
# 7a. LightGBM Feature Importance
# ──────────────────────────────────────────────
print("\n=== LightGBM Top 10 Most Important Features ===")
print_top_features(best_lgbm.feature_importances_)


# ══════════════════════════════════════════════
#  SAVE BEST PARAMS TO JSON
# ══════════════════════════════════════════════
params_dir = Path(__file__).resolve().parent.parent
lgbm_params_path = params_dir / "lgbm_best_params.json"

lgbm_save = lgbm_study.best_params.copy()
max_leaves = 2 ** lgbm_save["max_depth"]
if lgbm_save["num_leaves"] > max_leaves:
    lgbm_save["num_leaves"] = max_leaves

with open(lgbm_params_path, "w") as f:
    json.dump(lgbm_save, f, indent=2)

print(f"\nSaved LightGBM best params → {lgbm_params_path}")

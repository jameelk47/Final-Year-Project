import json
import numpy as np
import optuna
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
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
# 2. Custom scorer: R² in dollar-space
# ──────────────────────────────────────────────
def dollar_r2(y_true, y_pred):
    return r2_score(np.expm1(y_true), np.expm1(y_pred))

dollar_r2_scorer = make_scorer(dollar_r2)


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


# ──────────────────────────────────────────────
# 3. Feature name helper
# ──────────────────────────────────────────────
def get_feature_names():
    enc = fiverr_dss_pipeline.named_steps["encoding"]
    tfidf_features = enc.named_transformers_["tfidf"].get_feature_names_out()
    ohe_features = enc.named_transformers_["ohe"].get_feature_names_out(cat_col)
    target_features = np.array([f"{col}_te" for col in target_enc_col])
    num_features_arr = np.array(num_cols)
    return np.concatenate([tfidf_features, ohe_features, target_features, num_features_arr])


def print_top_features(importances, n=10):
    feature_names = get_feature_names()
    top_indices = np.argsort(importances)[-n:][::-1]
    print(f"\n=== Top {n} Most Important Features ===")
    for idx in top_indices:
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        print(f"  {name}: {importances[idx]:.4f}")


# ══════════════════════════════════════════════
#  RANDOM FOREST OPTIMISATION
# ══════════════════════════════════════════════

# ──────────────────────────────────────────────
# 4. Optuna objective
# ──────────────────────────────────────────────
def rf_objective(trial):
    params = {
        # Number of trees: more = better but slower; RF saturates early
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        # Max tree depth — controls bias/variance trade-off
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        # Min samples before an internal node can split further
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        # Min samples in each leaf — controls leaf granularity
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        # Fraction of features to consider at each split
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
        # Subsample rows without replacement at each tree (reduces correlation)
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
        "random_state": 42,
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X_train_proc, y_train, cv=kf,
        scoring=dollar_r2_scorer, n_jobs=-1,
    )
    return scores.mean()


# ──────────────────────────────────────────────
# 5. Run optimisation
# ──────────────────────────────────────────────
N_TRIALS = 100

rf_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(),
    study_name="random_forest",
)

rf_study.optimize(rf_objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\n=== Random Forest Optuna Results ===")
print(f"Best CV R² (dollar-space): {rf_study.best_value:.4f}")
print(f"Best params: {rf_study.best_params}")


# ──────────────────────────────────────────────
# 6. Train final model with best parameters
# ──────────────────────────────────────────────
best_rf_params = rf_study.best_params.copy()
best_rf_params.update({"random_state": 42, "n_jobs": -1})

best_rf = RandomForestRegressor(**best_rf_params)
best_rf.fit(X_train_proc, y_train)

evaluate_dollar_space(y_test, best_rf.predict(X_test_proc), "Tuned Random Forest")


# ──────────────────────────────────────────────
# 7. Full-data cross-validation (sanity check)
# ──────────────────────────────────────────────
cv_params = rf_study.best_params.copy()
cv_params.update({"random_state": 42, "n_jobs": -1})

rf_pipeline = Pipeline([
    ("preprocess", fiverr_dss_pipeline),
    ("model", RandomForestRegressor(**cv_params)),
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_cv_scores = cross_val_score(rf_pipeline, X, y, cv=kf, scoring=dollar_r2_scorer)

print("\n=== Random Forest 5-Fold CV on Full Data (dollar-space) ===")
print("Fold R² scores:", np.round(rf_cv_scores, 4))
print(f"Mean R²: {rf_cv_scores.mean():.4f}")
print(f"Std  R²: {rf_cv_scores.std():.4f}")


# ──────────────────────────────────────────────
# 8. Feature importance
# ──────────────────────────────────────────────
print_top_features(best_rf.feature_importances_)


# ──────────────────────────────────────────────
# 9. Save best params to JSON
# ──────────────────────────────────────────────
params_dir = Path(__file__).resolve().parent.parent
rf_params_path = params_dir / "rf_best_params.json"

with open(rf_params_path, "w") as f:
    json.dump(rf_study.best_params, f, indent=2)

print(f"\nSaved Random Forest best params -> {rf_params_path}")

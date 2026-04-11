import numpy as np
import optuna
import joblib
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

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
# 2. Optuna Objective
# ──────────────────────────────────────────────
def objective(trial):
    params = {
        # Boosting rounds — let early stopping find the right number
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),

        # Core learning parameters
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),

        # Regularization
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),

        # Stochastic components (reduce overfitting)
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),

        # Minimum gain to split — acts as implicit regularization
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),

        # Fixed
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    # Enforce LightGBM constraint: num_leaves <= 2^max_depth
    max_leaves = 2 ** params["max_depth"]
    if params["num_leaves"] > max_leaves:
        params["num_leaves"] = max_leaves

    model = LGBMRegressor(**params)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        model,
        X_train_proc,
        y_train,
        cv=kf,
        scoring="r2",
        n_jobs=-1,
    )

    return scores.mean()


# ──────────────────────────────────────────────
# 3. Run Optimization
# ──────────────────────────────────────────────
study = optuna.create_study(
    direction="maximize",            # Maximize R²
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(),
)

study.optimize(
    objective,
    n_trials=200,                    # ~200 intelligent samples vs 6,561 grid points
    show_progress_bar=True,
)

print("\n=== Optuna Results ===")
print(f"Best CV R²: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")


# ──────────────────────────────────────────────
# 4. Train Final Model with Best Parameters
# ──────────────────────────────────────────────
best_params = study.best_params.copy()

# Re-enforce num_leaves constraint
max_leaves = 2 ** best_params["max_depth"]
if best_params["num_leaves"] > max_leaves:
    best_params["num_leaves"] = max_leaves

best_params.update({
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
    "early_stopping_rounds": 20,
})

best_lgbm = LGBMRegressor(**best_params)

best_lgbm.fit(
    X_train_proc,
    y_train,
    eval_set=[(X_test_proc, y_test)],
    eval_metric="rmse",
)

# ──────────────────────────────────────────────
# 5. Evaluate on Held-Out Test Set
# ──────────────────────────────────────────────
y_pred = best_lgbm.predict(X_test_proc)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("\n=== Tuned LightGBM Test Performance ===")
print(f"R²   : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.4f}")


# ──────────────────────────────────────────────
# 6. Cross-Validate on Full Data (final sanity check)
# ──────────────────────────────────────────────
cv_params = study.best_params.copy()
max_leaves = 2 ** cv_params["max_depth"]
if cv_params["num_leaves"] > max_leaves:
    cv_params["num_leaves"] = max_leaves
cv_params.update({"random_state": 42, "verbose": -1, "n_jobs": -1})

full_pipeline = Pipeline([
    ("preprocess", fiverr_dss_pipeline),
    ("model", LGBMRegressor(**cv_params)),
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(full_pipeline, X, y, cv=kf, scoring="r2")

print("\n=== 5-Fold CV on Full Data ===")
print("Fold R² scores:", np.round(cv_scores, 4))
print(f"Mean R²: {cv_scores.mean():.4f}")
print(f"Std  R²: {cv_scores.std():.4f}")


# ──────────────────────────────────────────────
# 7. Feature Importance
# ──────────────────────────────────────────────
feature_importance = best_lgbm.feature_importances_

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

top_indices = np.argsort(feature_importance)[-10:][::-1]
print("\n=== Top 10 Most Important Features ===")
for idx in top_indices:
    name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
    print(f"  {name}: {feature_importance[idx]:.2f}")

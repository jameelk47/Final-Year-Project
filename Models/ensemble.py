import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

# Import fitted preprocessing pipeline, full data, train/test splits, and column definitions
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

# 1. Transform features using the fitted preprocessing pipeline
X_train_proc = fiverr_dss_pipeline.transform(X_train)
X_test_proc = fiverr_dss_pipeline.transform(X_test)

# LightGBM can handle sparse matrices natively, but we'll convert to dense for consistency
X_train_proc = X_train_proc.toarray()
X_test_proc = X_test_proc.toarray()

print("Transformed train shape:", X_train_proc.shape)
print("Transformed test shape:", X_test_proc.shape)

# 2. Create and train LightGBM model
lgbm = LGBMRegressor(
    n_estimators=453,
    learning_rate=0.011639982228003031,
    max_depth=12,
    num_leaves=94,
    min_child_samples=5,
    subsample=0.7526087090577469,
    colsample_bytree=0.5577305488227938,
    reg_alpha=0.0010019288026431733,
    reg_lambda=4.884203538265408,
    min_split_gain=0.259428302991762,
    random_state=42

)

lgbm.fit(
    X_train_proc,
    y_train,
    eval_set=[(X_test_proc, y_test)],
    eval_metric="rmse",
)

# 3. Predict on the test set
y_pred = lgbm.predict(X_test_proc)

# 4. Evaluation metrics: R^2, MAE, MAPE
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("\n=== LightGBM Performance ===")
print(f"R^2   : {r2:.4f}")
print(f"MAE   : {mae:.4f}")
print(f"MAPE  : {mape:.4f}")

# 5. K-fold cross-validation on full pipeline (preprocessing + LightGBM)
print("\n=== 5-Fold Cross-Validation (R^2) ===")

cv_lgbm = LGBMRegressor(
    n_estimators=453,
    learning_rate=0.011639982228003031,
    max_depth=12,
    num_leaves=94,
    min_child_samples=5,
    subsample=0.7526087090577469,
    colsample_bytree=0.5577305488227938,
    reg_alpha=0.0010019288026431733,
    reg_lambda=4.884203538265408,
    min_split_gain=0.259428302991762,
    random_state=42,
)

full_pipeline = Pipeline(
    steps=[
        ("preprocess", fiverr_dss_pipeline),
        ("model", cv_lgbm),
    ]
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(full_pipeline, X, y, cv=kf, scoring="r2")

print("Fold R^2 scores:", np.round(cv_scores, 4))
print("Mean R^2:", cv_scores.mean().round(4))

# 6. Feature importance (top 10) with feature names from the pipeline
feature_importance = lgbm.feature_importances_

# Recover feature names from the fitted ColumnTransformer
enc = fiverr_dss_pipeline.named_steps["encoding"]

tfidf = enc.named_transformers_["tfidf"]
ohe = enc.named_transformers_["ohe"]
target_enc = enc.named_transformers_["target_enc"]  # noqa: F841 - used for naming
# Note: numeric columns go through 'num_pipeline' transformer, but we use num_cols for names

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
    if idx < len(feature_names):
        name = feature_names[idx]
    else:
        name = f"feature_{idx}"
    print(f"{name}: {feature_importance[idx]:.2f}")

joblib.dump(lgbm, 'lgbm_model.pkl')
joblib.dump(fiverr_dss_pipeline, 'preprocessor.pkl')
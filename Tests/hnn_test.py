import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

# Import fitted preprocessing pipeline and data splits
from Dataset.preprocessing import fiverr_dss_pipeline, X_train, X_test, y_train, y_test

# Import the heteroscedastic neural network regressor
from Models.hnn import HeteroscedasticKerasRegressor

# 1. Transform features using the fitted preprocessing pipeline
X_train_proc = fiverr_dss_pipeline.transform(X_train)
X_test_proc = fiverr_dss_pipeline.transform(X_test)

# The column transformer returns sparse CSR matrices; convert to dense arrays
X_train_proc = X_train_proc.toarray()
X_test_proc = X_test_proc.toarray()

print("Transformed train shape (dense):", X_train_proc.shape)
print("Transformed test shape (dense):", X_test_proc.shape)

# 2. Create and train the heteroscedastic model
hnn = HeteroscedasticKerasRegressor(
    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    verbose=1,        # set to 0 to silence training logs
)

hnn.fit(X_train_proc, y_train)

# 3. Predict on the test set (with uncertainty)
y_pred_mean, y_pred_std = hnn.predict(X_test_proc, return_std=True)

# 4. Evaluation metrics: R^2, MAE, MAPE
r2 = r2_score(y_test, y_pred_mean)
mae = mean_absolute_error(y_test, y_pred_mean)
mape = mean_absolute_percentage_error(y_test, y_pred_mean)

print("\n=== Heteroscedastic NN Performance ===")
print(f"R^2   : {r2:.4f}")
print(f"MAE   : {mae:.4f}")
print(f"MAPE  : {mape:.4f}")

# 5. Optional: inspect a few predictions with uncertainty
for i in range(5):
    print(
        f"True: {y_test.iloc[i]:.3f}, "
        f"Pred: {y_pred_mean[i]:.3f}, "
        f"Std (uncertainty): {y_pred_std[i]:.3f}"
    )
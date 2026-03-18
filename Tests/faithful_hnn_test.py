import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

# Import fitted preprocessing pipeline and data splits
from Dataset.preprocessing import fiverr_dss_pipeline, X_train, X_test, y_train, y_test

# Import the faithful heteroscedastic neural network regressor
from Models.faithful_hnn import FaithfulHeteroscedasticRegressor

# 1. Transform features using the fitted preprocessing pipeline
X_train_proc = fiverr_dss_pipeline.transform(X_train)
X_test_proc = fiverr_dss_pipeline.transform(X_test)

# The column transformer returns sparse CSR matrices; convert to dense arrays
X_train_proc = X_train_proc.toarray()
X_test_proc = X_test_proc.toarray()

print("Transformed train shape (dense):", X_train_proc.shape)
print("Transformed test shape (dense):", X_test_proc.shape)

# 2. Create and train the faithful heteroscedastic model
faithful_hnn = FaithfulHeteroscedasticRegressor(
    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    verbose=0,        # set to 1 to see training progress (if implemented)
)

faithful_hnn.fit(X_train_proc, y_train)

# 3. Predict on the test set (with uncertainty)
y_pred_mean, y_pred_std = faithful_hnn.predict(X_test_proc, return_std=True)

# 4. Evaluation metrics: R^2, MAE, MAPE
r2 = r2_score(y_test, y_pred_mean)
mae = mean_absolute_error(y_test, y_pred_mean)
mape = mean_absolute_percentage_error(y_test, y_pred_mean)

print("\n=== Faithful Heteroscedastic NN Performance ===")
print(f"R^2   : {r2:.4f}")
print(f"MAE   : {mae:.4f}")
print(f"MAPE  : {mape:.4f}")

# 5. Optional: inspect a few predictions with uncertainty
print("\n=== Sample Predictions with Uncertainty ===")
for i in range(5):
    print(
        f"True: {y_test.iloc[i]:.3f}, "
        f"Pred: {y_pred_mean[i]:.3f}, "
        f"Std (uncertainty): {y_pred_std[i]:.3f}"
    )

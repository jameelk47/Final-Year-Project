import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
# Import your custom HNN class so joblib can reconstruct the object
from Models.hnn import HeteroscedasticKerasRegressor 
from Dataset.preprocessing import X_test, y_test 
import matplotlib.pyplot as plt
import seaborn as sns

def run_calibration():
    # 1. Load the same assets your Predictor uses
    print("Loading models and data...")
    pipeline = joblib.load('preprocessor.pkl')
    lgbm = joblib.load('lgbm_model.pkl')
    hnn = joblib.load('hnn_wrapper.pkl')
    hnn.model_ = tf.keras.models.load_model('hnn_weights.keras', compile=False)

    # 2. Load your TEST data (the 20% the model hasn't 'memorized')
    # Assuming your test data is saved or accessible via your Dataset module
    
    # 3. Process the test data
    X_test_proc = pipeline.transform(X_test).toarray()
    
    # 4. Generate Predictions
    print("Generating ensemble predictions for calibration...")
    lgbm_preds = lgbm.predict(X_test_proc)
    hnn_mu, hnn_sigma = hnn.predict(X_test_proc, return_std=True)
    
    # 5. Calculate Metrics
    disagreement = np.abs(lgbm_preds - hnn_mu)
    errors = np.abs(y_test.values - lgbm_preds)  # Per-sample absolute error

    # 6. Risk-Coverage Analysis
    # Sweep through percentiles to find where error starts spiking
    percentiles = np.arange(50, 100, 1)

    # --- Sigma (confidence) curve ---
    sigma_thresholds, sigma_errors, sigma_coverages = [], [], []
    for p in percentiles:
        thresh = np.percentile(hnn_sigma, p)
        mask = hnn_sigma <= thresh
        if mask.sum() == 0:
            continue
        sigma_thresholds.append(thresh)
        sigma_errors.append(errors[mask].mean())
        sigma_coverages.append(mask.mean())

    # --- Divergence curve ---
    div_thresholds, div_errors, div_coverages = [], [], []
    for p in percentiles:
        thresh = np.percentile(disagreement, p)
        mask = disagreement <= thresh
        if mask.sum() == 0:
            continue
        div_thresholds.append(thresh)
        div_errors.append(errors[mask].mean())
        div_coverages.append(mask.mean())

    # 7. Plot Risk vs Coverage
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(sigma_coverages, sigma_errors, 'b-o', markersize=3)
    axes[0].set_xlabel('Coverage (% of predictions kept)')
    axes[0].set_ylabel('MAE on kept predictions')
    axes[0].set_title('Risk vs Coverage — HNN Sigma')
    axes[0].axhline(np.mean(errors), color='red', linestyle='--', label='Overall MAE')
    axes[0].legend()

    axes[1].plot(div_coverages, div_errors, 'g-o', markersize=3)
    axes[1].set_xlabel('Coverage (% of predictions kept)')
    axes[1].set_ylabel('MAE on kept predictions')
    axes[1].set_title('Risk vs Coverage — Model Divergence')
    axes[1].axhline(np.mean(errors), color='red', linestyle='--', label='Overall MAE')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # 8. Find the "elbow" (where the error gradient spikes)
    sigma_gradient = np.gradient(sigma_errors)
    sigma_elbow_idx = np.argmax(sigma_gradient > np.mean(sigma_gradient) + np.std(sigma_gradient))

    div_gradient = np.gradient(div_errors)
    div_elbow_idx = np.argmax(div_gradient > np.mean(div_gradient) + np.std(div_gradient))

    optimal_sigma = sigma_thresholds[sigma_elbow_idx]
    optimal_div = div_thresholds[div_elbow_idx]

    print("\n" + "=" * 45)
    print("OPTIMAL THRESHOLDS (Risk-Coverage Elbow)")
    print("=" * 45)
    print(f"Sigma threshold (GREEN/YELLOW):     {optimal_sigma:.3f}")
    print(f"Divergence threshold (GREEN/RED):    {optimal_div:.3f}")
    print("-" * 45)
    print(f"Coverage at sigma elbow:             {sigma_coverages[sigma_elbow_idx]:.1%}")
    print(f"Coverage at divergence elbow:        {div_coverages[div_elbow_idx]:.1%}")
    print("=" * 45)
    print("\nACTION: Update your UncertaintyGater with these values.")

    # Return models and processed test set for further EDA / analysis
    return lgbm, hnn, X_test_proc


def run_uncertainty_eda(lgbm_model, hnn_model, X_test_proc):
    # 1. Get Predictions
    lgbm_mu = lgbm_model.predict(X_test_proc)
    hnn_mu, hnn_sigma = hnn_model.predict(X_test_proc, return_std=True)
    
    # 2. Calculate Divergence
    divergence = np.abs(lgbm_mu - hnn_mu)
    
    # 3. Create a Diagnostic DataFrame
    df_eda = pd.DataFrame({
        'hnn_sigma': hnn_sigma,
        'divergence': divergence
    })

    # 4. Plot Distributions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df_eda['hnn_sigma'], kde=True, color='blue')
    plt.title('Distribution of HNN Sigma (Market Noise)')
    plt.axvline(df_eda['hnn_sigma'].mean(), color='red', linestyle='--', label='Mean')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df_eda['divergence'], kde=True, color='green')
    plt.title('Distribution of Model Divergence (System Conflict)')
    plt.axvline(df_eda['divergence'].mean(), color='red', linestyle='--', label='Mean')
    
    plt.legend()
    plt.show()

    # 5. Print Percentile Table for your Thesis
    print("--- Percentile Breakdown ---")
    print(df_eda.describe(percentiles=[.5, .75, .9, .95, .99]))

# Run this using your loaded models and processed test set


if __name__ == "__main__":
    lgbm, hnn, X_test_proc = run_calibration()
    run_uncertainty_eda(lgbm, hnn, X_test_proc)
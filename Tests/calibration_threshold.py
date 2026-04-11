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

    elbow_sigma = sigma_thresholds[sigma_elbow_idx]
    elbow_div = div_thresholds[div_elbow_idx]

    # 9. Percentile-based threshold candidates
    GREEN_PCTL, YELLOW_PCTL, DIV_PCTL = 30, 70, 70

    green_sigma = np.percentile(hnn_sigma, GREEN_PCTL)
    yellow_sigma = np.percentile(hnn_sigma, YELLOW_PCTL)
    div_pctl_val = np.percentile(disagreement, DIV_PCTL)

    # MAE for predictions within each proposed tier
    green_mask = hnn_sigma <= green_sigma
    yellow_mask = (hnn_sigma > green_sigma) & (hnn_sigma <= yellow_sigma)
    red_vol_mask = hnn_sigma > yellow_sigma

    green_mae = errors[green_mask].mean() if green_mask.sum() > 0 else float('nan')
    yellow_mae = errors[yellow_mask].mean() if yellow_mask.sum() > 0 else float('nan')
    red_vol_mae = errors[red_vol_mask].mean() if red_vol_mask.sum() > 0 else float('nan')
    overall_mae = errors.mean()

    # 10. Report
    print("\n" + "=" * 60)
    print("  RISK-COVERAGE ELBOW (reference)")
    print("=" * 60)
    print(f"  Sigma elbow:       {elbow_sigma:.4f}  (coverage {sigma_coverages[sigma_elbow_idx]:.1%})")
    print(f"  Divergence elbow:  {elbow_div:.4f}  (coverage {div_coverages[div_elbow_idx]:.1%})")

    print("\n" + "=" * 60)
    print("  PERCENTILE-BASED THRESHOLDS")
    print("=" * 60)
    print(f"  green_threshold  (sigma  P{GREEN_PCTL}):   {green_sigma:.4f}")
    print(f"  yellow_threshold (sigma  P{YELLOW_PCTL}):   {yellow_sigma:.4f}")
    print(f"  divergence_threshold (P{DIV_PCTL}):   {div_pctl_val:.4f}")

    print("\n" + "-" * 60)
    print("  CROSS-VALIDATION: MAE per tier")
    print("-" * 60)
    print(f"  GREEN  (sigma <= {green_sigma:.4f}):  MAE = {green_mae:.4f}  ({green_mask.sum()} gigs)")
    print(f"  YELLOW ({green_sigma:.4f} < sigma <= {yellow_sigma:.4f}):  MAE = {yellow_mae:.4f}  ({yellow_mask.sum()} gigs)")
    print(f"  RED    (sigma > {yellow_sigma:.4f}):  MAE = {red_vol_mae:.4f}  ({red_vol_mask.sum()} gigs)")
    print(f"  Overall MAE:                    {overall_mae:.4f}")

    # Validation check: percentile thresholds vs elbow
    print("\n" + "-" * 60)
    print("  VALIDATION")
    print("-" * 60)
    if green_sigma <= elbow_sigma:
        print(f"  ✓ green_threshold ({green_sigma:.4f}) is at or below sigma elbow ({elbow_sigma:.4f}) — safe")
    else:
        print(f"  ✗ green_threshold ({green_sigma:.4f}) is ABOVE sigma elbow ({elbow_sigma:.4f}) — consider lowering")

    if yellow_sigma <= elbow_sigma:
        print(f"  ! yellow_threshold ({yellow_sigma:.4f}) is below sigma elbow — very conservative")
    elif yellow_sigma <= elbow_sigma * 1.5:
        print(f"  ✓ yellow_threshold ({yellow_sigma:.4f}) is near sigma elbow ({elbow_sigma:.4f}) — reasonable")
    else:
        print(f"  ⚠ yellow_threshold ({yellow_sigma:.4f}) is well above sigma elbow ({elbow_sigma:.4f}) — check tier MAE")

    if div_pctl_val <= elbow_div:
        print(f"  ✓ divergence_threshold ({div_pctl_val:.4f}) is at or below elbow ({elbow_div:.4f}) — safe")
    else:
        print(f"  ⚠ divergence_threshold ({div_pctl_val:.4f}) is above elbow ({elbow_div:.4f}) — check tier MAE")
    print("=" * 60)

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
    print(df_eda.describe(percentiles=[.25, .30, .50, .70, .75, .85, .90, .95, .99]))

# Run this using your loaded models and processed test set


if __name__ == "__main__":
    lgbm, hnn, X_test_proc = run_calibration()
    run_uncertainty_eda(lgbm, hnn, X_test_proc)
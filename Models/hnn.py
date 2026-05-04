import json
import random
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from Dataset.preprocessing import fiverr_dss_pipeline, X_train, X_test, y_train, y_test
import matplotlib.pyplot as plt
import joblib

# ──────────────────────────────────────────────
# Load best params (from Tests/hnn_optimization.py)
# ──────────────────────────────────────────────

# Keys written by Optuna for bookkeeping — not model hyperparameters.
_HNN_JSON_METADATA = frozenset({
    "objective",
    "best_objective_value",
    "best_nll",
    "best_val_nll",
    "best_val_nll_gaussian",
    "best_val_r2_dollar",
    "best_val_mae_dollar",
    "use_composite_objective",
    "composite_mae_log_weight",
    "narrow_search_space",
})

params_path = Path(__file__).resolve().parent.parent / "hnn_best_params.json"
with open(params_path) as f:
    _raw_params = json.load(f)

_training_params = {
    "learning_rate": _raw_params["learning_rate"],
    "batch_size":    int(_raw_params["batch_size"]),
}
_arch_params = {
    k: v for k, v in _raw_params.items()
    if k not in _HNN_JSON_METADATA and k not in ("learning_rate", "batch_size")
}
_arch_params["n_layers"] = int(_arch_params["n_layers"])

# Used by the legacy aleatoric_loss function below.
BETA = _arch_params["beta"]


@tf.keras.saving.register_keras_serializable(package="CustomHNN")
class AleatoricLoss(tf.keras.losses.Loss):
    """Heteroscedastic Gaussian NLL with variance stabilisation (beta), serialisable for save/load."""

    def __init__(self, beta=0.5, name="aleatoric_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.beta = float(beta)

    def get_config(self):
        cfg = super().get_config()
        cfg["beta"] = self.beta
        return cfg

    def call(self, y_true, y_pred):
        mu = y_pred[:, :1]
        log_var = y_pred[:, 1:]
        nll = 0.5 * K.exp(-log_var) * K.square(y_true - mu) + 0.5 * log_var
        weight = tf.stop_gradient(K.exp(log_var)) ** self.beta
        return tf.reduce_mean(weight * nll)


def build_hnn_model(input_dim, arch, learning_rate=0.001, random_state=None):
    """Build HNN from architecture dict (n_layers, units_*, dropout_*, batchnorm_*, beta)."""
    if random_state is not None:
        tf.random.set_seed(random_state)

    n_layers = int(arch["n_layers"])
    loss = AleatoricLoss(beta=arch["beta"])

    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for i in range(n_layers):
        x = layers.Dense(int(arch[f"units_{i}"]), activation="relu")(x)
        if arch.get(f"batchnorm_{i}", False):
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(float(arch[f"dropout_{i}"]))(x, training=True)

    outputs = layers.Dense(2)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
    )
    return model


class HeteroscedasticKerasRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, epochs=100, n_samples=50, verbose=0, random_state=42):
        self.epochs = epochs
        self.n_samples = n_samples
        self.verbose = verbose
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
            tf.random.set_seed(self.random_state)
            tf.keras.utils.set_random_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        input_dim = X.shape[1]
        self.model_ = build_hnn_model(
            input_dim,
            _arch_params,
            learning_rate=_training_params["learning_rate"],
            random_state=self.random_state,
        )

        es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        self.model_.fit(
            X,
            y,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=_training_params["batch_size"],
            verbose=self.verbose,
            callbacks=[es],
        )
        return self

    def predict(self, X, return_std=False, return_components=False):
        X = np.asarray(X, dtype=np.float32)

        mus = []
        log_vars = []

        for _ in range(self.n_samples):
            preds = self.model_.predict(X, verbose=0)
            mus.append(preds[:, 0])
            log_vars.append(preds[:, 1])

        mus = np.array(mus)
        log_vars = np.array(log_vars)

        final_mu = np.mean(mus, axis=0)

        if return_std:
            aleatoric_var = np.mean(np.exp(log_vars), axis=0)
            epistemic_var = np.var(mus, axis=0)
            total_std = np.sqrt(aleatoric_var + epistemic_var)
            if return_components:
                aleatoric_std = np.sqrt(aleatoric_var)
                epistemic_std = np.sqrt(epistemic_var)
                return final_mu, total_std, aleatoric_std, epistemic_std
            return final_mu, total_std

        return final_mu


if __name__ == "__main__":
    X_train_proc = fiverr_dss_pipeline.transform(X_train)
    X_test_proc = fiverr_dss_pipeline.transform(X_test)

    X_train_proc = X_train_proc.toarray()
    X_test_proc = X_test_proc.toarray()

    print("Transformed train shape (dense):", X_train_proc.shape)
    print("Transformed test shape (dense):", X_test_proc.shape)

    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    hnn = HeteroscedasticKerasRegressor(
        epochs=100,
        verbose=1,
        random_state=RANDOM_STATE,
    )

    hnn.fit(X_train_proc, y_train)

    y_pred_mean, y_pred_std, y_aleatoric_std, y_epistemic_std = hnn.predict(
        X_test_proc, return_std=True, return_components=True
    )

    y_test_dollars = np.expm1(y_test)
    y_pred_dollars = np.expm1(y_pred_mean)

    r2 = r2_score(y_test_dollars, y_pred_dollars)
    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
    mape = mean_absolute_percentage_error(y_test_dollars, y_pred_dollars)
    rmse = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))

    print("\n=== Heteroscedastic NN Performance (dollar-space) ===")
    print(f"R²    : {r2:.4f}")
    print(f"MAE   : ${mae:.2f}")
    print(f"MAPE  : {mape:.4f}")
    print(f"RMSE  : ${rmse:.2f}")

    y_test_arr = np.asarray(y_test)
    variance = y_pred_std ** 2
    nll = 0.5 * np.log(2 * np.pi * variance) + 0.5 * ((y_test_arr - y_pred_mean) ** 2 / variance)
    print(f"\n=== Gaussian NLL (log-space) ===")
    print(f"Mean NLL : {np.mean(nll):.4f}")
    print(f"Median NLL: {np.median(nll):.4f}")

    abs_errors = np.abs(y_test_arr - y_pred_mean)
    n_bins = 10
    sorted_idx = np.argsort(y_pred_std)
    bin_size = len(sorted_idx) // n_bins

    bin_total_unc = []
    bin_aleatoric_unc = []
    bin_epistemic_unc = []
    bin_errors = []

    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_idx)
        idx = sorted_idx[start:end]
        bin_total_unc.append(np.mean(y_pred_std[idx]))
        bin_aleatoric_unc.append(np.mean(y_aleatoric_std[idx]))
        bin_epistemic_unc.append(np.mean(y_epistemic_std[idx]))
        bin_errors.append(np.mean(abs_errors[idx]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    plots = [
        (bin_total_unc, "Total Uncertainty", "#1f77b4"),
        (bin_aleatoric_unc, "Aleatoric Uncertainty", "#ff7f0e"),
        (bin_epistemic_unc, "Epistemic Uncertainty", "#2ca02c"),
    ]

    for ax, (x_vals, title, color) in zip(axes, plots):
        ax.plot(x_vals, bin_errors, "o-", linewidth=2, markersize=5, color=color)
        ax.set_xlabel("Mean Predicted Uncertainty (σ)")
        ax.set_title(f"Error vs {title}")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean Absolute Error (log-space)")
    fig.suptitle("HNN Uncertainty Components vs Error", y=1.03)
    plt.tight_layout()
    plt.savefig("hnn_error_vs_uncertainty.png", dpi=150)
    plt.show()
    print("Saved plot → hnn_error_vs_uncertainty.png")

    for i in range(5):
        print(
            f"True: ${np.expm1(y_test.iloc[i]):.2f}, "
            f"Pred: ${np.expm1(y_pred_mean[i]):.2f}, "
            f"Std (uncertainty): {y_pred_std[i]:.3f}"
        )

    hnn.model_.save("hnn_weights.keras")
    joblib.dump(hnn, "hnn_wrapper.pkl")

import json
import random
import numpy as np
import optuna
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from optuna.trial import TrialState
from pathlib import Path

from Dataset.preprocessing import (
    fiverr_dss_pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
)

# ── Optimisation policy ───────────────────────────────────────────
# Pure val NLL can pick high-sigma / odd architectures that look good on the
# validation slice but hurt test R². A composite score ties likelihood to point fit.
USE_COMPOSITE_OBJECTIVE = True
# Weight on mean absolute error in log-price (same scale as targets); ~0.25–0.40 typical.
COMPOSITE_MAE_LOG_WEIGHT = 0.32

# If True, keep architectures closer to Models/hnn.py (3 dense layers, moderate dropout).
NARROW_SEARCH_SPACE = True

# ──────────────────────────────────────────────
# 1. Transform features
# ──────────────────────────────────────────────
X_train_proc = fiverr_dss_pipeline.transform(X_train).toarray()
X_test_proc = fiverr_dss_pipeline.transform(X_test).toarray()

print("Transformed train shape:", X_train_proc.shape)
print("Transformed test shape:", X_test_proc.shape)

RANDOM_STATE = 42
INPUT_DIM = X_train_proc.shape[1]

# Fixed train / val split (matches Keras: last 20% of arrays, no shuffle on split order)
_n = len(X_train_proc)
_n_val = max(1, int(_n * 0.2))
X_tr_fit = X_train_proc[:-_n_val]
X_va_fit = X_train_proc[-_n_val:]
y_tr_fit = y_train.iloc[:-_n_val].values.astype(np.float32).reshape(-1, 1)
y_va_fit = y_train.iloc[-_n_val:].values.astype(np.float32)
print(f"Train rows for HPO: {len(X_tr_fit)}, validation rows: {len(X_va_fit)}")


# ──────────────────────────────────────────────
# 2. Flexible model builder (architecture is part of the search)
# ──────────────────────────────────────────────
def build_trial_model(input_dim, params):
    """Build an HNN with architecture + loss params from an Optuna trial."""
    tf.random.set_seed(RANDOM_STATE)

    inputs = layers.Input(shape=(input_dim,))
    x = inputs

    for i in range(params["n_layers"]):
        x = layers.Dense(params[f"units_{i}"], activation="relu")(x)
        if params[f"batchnorm_{i}"]:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(params[f"dropout_{i}"])(x, training=True)

    outputs = layers.Dense(2)(x)

    beta = params["beta"]

    @tf.keras.saving.register_keras_serializable()
    def weighted_nll(y_true, y_pred):
        mu = y_pred[:, :1]
        log_var = y_pred[:, 1:]
        nll = 0.5 * K.exp(-log_var) * K.square(y_true - mu) + 0.5 * log_var
        weight = tf.stop_gradient(K.exp(log_var)) ** beta
        return K.mean(weight * nll)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss=weighted_nll,
    )
    return model


# ──────────────────────────────────────────────
# 3. Gaussian NLL scorer (unweighted, for monitoring)
# ──────────────────────────────────────────────
def gaussian_nll(y_true, mu, sigma):
    """Standard Gaussian NLL — used for evaluation, not training."""
    var = sigma ** 2
    return float(np.mean(0.5 * np.log(2 * np.pi * var)
                         + 0.5 * (y_true - mu) ** 2 / var))


def mc_predict(model, X, n_samples=30):
    """MC-dropout forward passes → (mean_mu, mean_sigma)."""
    mus, log_vars = [], []
    for _ in range(n_samples):
        preds = model.predict(X, verbose=0)
        mus.append(preds[:, 0])
        log_vars.append(preds[:, 1])
    mus = np.array(mus)
    log_vars = np.array(log_vars)
    mean_mu = np.mean(mus, axis=0)
    aleatoric_var = np.mean(np.exp(log_vars), axis=0)
    epistemic_var = np.var(mus, axis=0)
    total_sigma = np.sqrt(aleatoric_var + epistemic_var)
    return mean_mu, total_sigma


# ──────────────────────────────────────────────
# 4. Optuna objective
# ──────────────────────────────────────────────
N_MC_SAMPLES = 20   # fewer than production (50) to keep trials fast
MAX_EPOCHS = 120
PATIENCE = 15

def hnn_objective(trial):
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    if NARROW_SEARCH_SPACE:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1.5e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "beta": trial.suggest_float("beta", 0.35, 0.65),
            "n_layers": trial.suggest_int("n_layers", 3, 4),
        }
        _unit_choices = ([256, 512], [128, 256], [64, 128], [64, 128])
    else:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "beta": trial.suggest_float("beta", 0.0, 1.0),
            "n_layers": trial.suggest_int("n_layers", 2, 4),
        }
        _unit_choices = None

    for i in range(params["n_layers"]):
        if NARROW_SEARCH_SPACE:
            params[f"units_{i}"] = trial.suggest_categorical(
                f"units_{i}", list(_unit_choices[i])
            )
            params[f"dropout_{i}"] = trial.suggest_float(f"dropout_{i}", 0.08, 0.28)
        else:
            params[f"units_{i}"] = trial.suggest_categorical(
                f"units_{i}", [64, 128, 256, 512]
            )
            params[f"dropout_{i}"] = trial.suggest_float(f"dropout_{i}", 0.0, 0.45)
        params[f"batchnorm_{i}"] = trial.suggest_categorical(f"batchnorm_{i}", [True, False])

    # Fill unused layer slots so build_trial_model doesn't KeyError
    for i in range(params["n_layers"], 4):
        params[f"units_{i}"] = 64
        params[f"dropout_{i}"] = 0.0
        params[f"batchnorm_{i}"] = False

    model = build_trial_model(INPUT_DIM, params)

    # Pruning callback: report val_loss at each epoch
    class OptunaReportCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            val_loss = logs.get("val_loss")
            if val_loss is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

    model.fit(
        X_tr_fit,
        y_tr_fit,
        validation_data=(X_va_fit, y_va_fit.reshape(-1, 1)),
        epochs=MAX_EPOCHS,
        batch_size=params["batch_size"],
        verbose=0,
        callbacks=[es, OptunaReportCallback()],
    )

    # Optimise on validation NLL only — test set reserved for final evaluation
    mu, sigma = mc_predict(model, X_va_fit, n_samples=N_MC_SAMPLES)
    nll = gaussian_nll(y_va_fit, mu, sigma)

    y_pred_d = np.expm1(mu)
    y_true_d = np.expm1(y_va_fit)
    mae_dollar = float(mean_absolute_error(y_true_d, y_pred_d))
    r2_dollar = float(r2_score(y_true_d, y_pred_d))
    mae_log = float(np.mean(np.abs(y_va_fit - mu)))
    trial.set_user_attr("val_nll_gaussian", nll)
    trial.set_user_attr("mae_log_val", mae_log)
    trial.set_user_attr("mae_dollar_val", mae_dollar)
    trial.set_user_attr("r2_dollar_val", r2_dollar)
    trial.set_user_attr("mean_sigma_val", float(np.mean(sigma)))

    if USE_COMPOSITE_OBJECTIVE:
        obj = nll + COMPOSITE_MAE_LOG_WEIGHT * mae_log
        trial.set_user_attr("composite_score", obj)
        return obj
    return nll


# ──────────────────────────────────────────────
# 5. Run optimisation
# ──────────────────────────────────────────────
N_TRIALS = 40

_objective_label = (
    f"composite (NLL + {COMPOSITE_MAE_LOG_WEIGHT} * MAE_log)"
    if USE_COMPOSITE_OBJECTIVE
    else "validation Gaussian NLL"
)
print(f"\nObjective: minimize {_objective_label}")
print(f"Search space: {'narrow (near production HNN)' if NARROW_SEARCH_SPACE else 'wide'}\n")

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    study_name="hnn",
)

study.optimize(hnn_objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\n" + "=" * 55)
print("  HNN OPTUNA RESULTS (validation set — test untouched)")
print("=" * 55)
print(f"  Best objective     : {study.best_value:.4f}  ({_objective_label})")
best_attrs = study.best_trial.user_attrs
print(f"  Val Gaussian NLL   : {best_attrs.get('val_nll_gaussian', float('nan')):.4f}")
print(f"  Val MAE (log)      : {best_attrs.get('mae_log_val', float('nan')):.4f}")
print(f"  Best params        : {study.best_params}")
print(f"  Val MAE (dollars)  : ${best_attrs['mae_dollar_val']:.2f}")
print(f"  Val R2  (dollars)  : {best_attrs['r2_dollar_val']:.4f}")
print(f"  Val mean sigma     : {best_attrs['mean_sigma_val']:.4f}")
print("=" * 55)
print(
    "  Note: val metrics are on one slice of train; many trials can make that\n"
    "  slice look easier than test. Compare test R² to baseline hnn.py (~0.04)."
)
print("=" * 55)


# ──────────────────────────────────────────────
# 6. Retrain final model with best params
# ──────────────────────────────────────────────
print("\nRetraining final model with best hyperparameters...")

best = study.best_params.copy()

n_layers = best["n_layers"]
final_params = {
    "learning_rate": best["learning_rate"],
    "batch_size": best["batch_size"],
    "beta": best["beta"],
    "n_layers": n_layers,
}
for i in range(n_layers):
    final_params[f"units_{i}"] = best[f"units_{i}"]
    final_params[f"dropout_{i}"] = best[f"dropout_{i}"]
    final_params[f"batchnorm_{i}"] = best[f"batchnorm_{i}"]
for i in range(n_layers, 4):
    final_params[f"units_{i}"] = 64
    final_params[f"dropout_{i}"] = 0.0
    final_params[f"batchnorm_{i}"] = False

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)

final_model = build_trial_model(INPUT_DIM, final_params)
es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
# Same (train, val) tensors as during trials — avoids any subtle split mismatch.
final_model.fit(
    X_tr_fit,
    y_tr_fit,
    validation_data=(X_va_fit, y_va_fit.reshape(-1, 1)),
    epochs=MAX_EPOCHS,
    batch_size=final_params["batch_size"],
    verbose=1,
    callbacks=[es],
)


# ──────────────────────────────────────────────
# 7. Final evaluation (production MC samples)
# ──────────────────────────────────────────────
N_MC_FINAL = 50
mu, sigma = mc_predict(final_model, X_test_proc, n_samples=N_MC_FINAL)

y_true_d = np.expm1(y_test.values)
y_pred_d = np.expm1(mu)

nll = gaussian_nll(y_test.values, mu, sigma)
r2 = r2_score(y_true_d, y_pred_d)
mae = mean_absolute_error(y_true_d, y_pred_d)
mape = mean_absolute_percentage_error(y_true_d, y_pred_d)

print("\n=== Final HNN Performance (dollar-space) ===")
print(f"R2    : {r2:.4f}")
print(f"MAE   : ${mae:.2f}")
print(f"MAPE  : {mape:.4f}")
print(f"\n=== Gaussian NLL (log-space) ===")
print(f"Mean NLL   : {nll:.4f}")
print(f"Mean sigma : {np.mean(sigma):.4f}")


# ──────────────────────────────────────────────
# 8. Save best params to JSON
# ──────────────────────────────────────────────
params_dir = Path(__file__).resolve().parent.parent
params_path = params_dir / "hnn_best_params.json"

save_params = {
    "learning_rate": final_params["learning_rate"],
    "batch_size": final_params["batch_size"],
    "beta": final_params["beta"],
    "n_layers": n_layers,
}
for i in range(n_layers):
    save_params[f"units_{i}"] = int(final_params[f"units_{i}"])
    save_params[f"dropout_{i}"] = round(final_params[f"dropout_{i}"], 6)
    save_params[f"batchnorm_{i}"] = final_params[f"batchnorm_{i}"]

save_params["objective"] = _objective_label
save_params["best_objective_value"] = round(study.best_value, 6)
save_params["best_val_nll_gaussian"] = round(float(best_attrs.get("val_nll_gaussian", study.best_value)), 6)
save_params["best_val_r2_dollar"] = round(best_attrs["r2_dollar_val"], 6)
save_params["best_val_mae_dollar"] = round(best_attrs["mae_dollar_val"], 2)
save_params["use_composite_objective"] = USE_COMPOSITE_OBJECTIVE
save_params["composite_mae_log_weight"] = COMPOSITE_MAE_LOG_WEIGHT
save_params["narrow_search_space"] = NARROW_SEARCH_SPACE

with open(params_path, "w") as f:
    json.dump(save_params, f, indent=2)

print(f"\nSaved HNN best params -> {params_path}")


# ──────────────────────────────────────────────
# 9. Top-5 trials summary (COMPLETE only — pruned trials have no user attrs)
# ──────────────────────────────────────────────
print(f"\n=== Top 5 Trials (by {_objective_label}) ===")
_complete = [
    t for t in study.trials
    if t.state == TrialState.COMPLETE and t.value is not None
]
sorted_trials = sorted(_complete, key=lambda t: t.value)
for rank, t in enumerate(sorted_trials[:5], 1):
    attrs = t.user_attrs
    nll_g = attrs.get("val_nll_gaussian")
    r2v = attrs.get("r2_dollar_val")
    mae_v = attrs.get("mae_dollar_val")
    sig_v = attrs.get("mean_sigma_val")
    nll_s = f"{nll_g:.4f}" if nll_g is not None else "n/a"
    r2_s = f"{r2v:.4f}" if r2v is not None else "n/a"
    mae_s = f"${mae_v:.2f}" if mae_v is not None else "n/a"
    sig_s = f"{sig_v:.4f}" if sig_v is not None else "n/a"
    print(
        f"  #{rank}  obj={t.value:.4f}  val_NLL={nll_s}  "
        f"val_R2={r2_s}  val_MAE={mae_s}  val_sigma={sig_s}"
    )

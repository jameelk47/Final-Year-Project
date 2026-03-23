import numpy as np
import random
from sklearn.base import BaseEstimator, RegressorMixin
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from Dataset.preprocessing import fiverr_dss_pipeline, X_train, X_test, y_train, y_test
import joblib


# ---------------------------------------------------------------------------
# Loss function kept at module level for serialisation / loading
# ---------------------------------------------------------------------------

@tf.keras.saving.register_keras_serializable()
def aleatoric_loss(y_true, y_pred):
    mu = y_pred[:, :1]
    log_var = y_pred[:, 1:]
    return K.mean(0.5 * K.exp(-log_var) * K.square(y_true - mu) + 0.5 * log_var)


# ---------------------------------------------------------------------------
# Stop-gradient layer (Proposal 2)
# ---------------------------------------------------------------------------

@tf.keras.saving.register_keras_serializable()
class StopGradientLayer(layers.Layer):
    """Forward pass unchanged; backward pass blocks all gradients.

    Inserted between the shared trunk and the variance head so that
    the variance loss cannot update any trunk parameters.
    """
    def call(self, inputs):
        return tf.stop_gradient(inputs)


# ---------------------------------------------------------------------------
# Custom model with Newton-scaled mean gradient (Proposal 1) and
# stop-gradient trunk isolation for variance (Proposal 2)
# ---------------------------------------------------------------------------

@tf.keras.saving.register_keras_serializable()
class HeteroscedasticModel(models.Model):

    def train_step(self, data):
        x, y_true = data
        y_true = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)            # (batch, 2)
            mu      = y_pred[:, :1]
            log_var = y_pred[:, 1:]

            # ---- Proposal 1: Newton-scaled mean loss ----
            # Multiply squared error by detached precision Σ⁻¹ = exp(-log_var)
            # so ∂L_mu/∂μ = Σ⁻¹(μ − y)  — the Newton direction.
            # stop_gradient on log_var prevents this weighting from creating
            # a spurious gradient path through the variance head.
            precision = tf.exp(-tf.stop_gradient(log_var))
            loss_mu = tf.reduce_mean(0.5 * precision * tf.square(y_true - mu))

            # ---- Variance-head loss (standard NLL terms that depend on Σ) ----
            # Detach μ so variance loss cannot push the mean head around.
            mu_detached = tf.stop_gradient(mu)
            loss_var = tf.reduce_mean(
                0.5 * tf.exp(-log_var) * tf.square(y_true - mu_detached)
                + 0.5 * log_var
            )

            total_loss = loss_mu + loss_var

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        for m in self.metrics:
            if m.name == "loss":
                m.update_state(total_loss)
        return {"loss": total_loss}

    def test_step(self, data):
        x, y_true = data
        y_true = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)

        y_pred  = self(x, training=False)
        mu      = y_pred[:, :1]
        log_var = y_pred[:, 1:]

        nll = tf.reduce_mean(
            0.5 * tf.exp(-log_var) * tf.square(y_true - mu) + 0.5 * log_var
        )

        for m in self.metrics:
            if m.name == "loss":
                m.update_state(nll)
        return {"loss": nll}


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_hnn_model(input_dim, learning_rate=0.001, random_state=None):
    if random_state is not None:
        tf.random.set_seed(random_state)

    # ---- Shared trunk (θ_z) ----
    inputs = layers.Input(shape=(input_dim,))
    z = layers.Dense(256, activation='relu', name='trunk_dense1')(inputs)
    z = layers.BatchNormalization(name='trunk_bn1')(z)
    z = layers.Dropout(0.2, name='trunk_drop1')(z, training=True)   # MC-Dropout
    z = layers.Dense(128, activation='relu', name='trunk_dense2')(z)
    z = layers.Dropout(0.1, name='trunk_drop2')(z, training=True)   # MC-Dropout
    z = layers.Dense(64, activation='relu', name='trunk_dense3')(z)

    # ---- Mean head (θ_μ) — receives live trunk gradients ----
    mu = layers.Dense(1, name='mu_head')(z)

    # ---- Variance head (θ_σ) — trunk gradients blocked (Proposal 2) ----
    z_stopped = StopGradientLayer(name='stop_grad_trunk')(z)
    v = layers.Dense(64, activation='relu', name='var_hidden1')(z_stopped)
    v = layers.Dense(32, activation='relu', name='var_hidden2')(v)
    log_var = layers.Dense(1, name='logvar_head')(v)

    # ---- Concatenate → (batch, 2) : [μ, log_var] ----
    outputs = layers.Concatenate(name='output_concat')([mu, log_var])

    model = HeteroscedasticModel(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=aleatoric_loss,
    )
    return model


# ---------------------------------------------------------------------------
# Scikit-learn compatible wrapper
# ---------------------------------------------------------------------------

class HeteroscedasticKerasRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, epochs=100, batch_size=32, learning_rate=0.001,
                 n_samples=50, verbose=0, random_state=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.n_samples = n_samples
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
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )

        es = EarlyStopping(monitor='val_loss', patience=15,
                           restore_best_weights=True)
        self.model_.fit(
            X, y,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=[es],
        )
        return self

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=np.float32)

        mus = []
        log_vars = []

        for _ in range(self.n_samples):
            preds = self.model_.predict(X, verbose=0)
            mus.append(preds[:, 0])
            log_vars.append(preds[:, 1])

        mus = np.array(mus)            # (n_samples, n_rows)
        log_vars = np.array(log_vars)  # (n_samples, n_rows)

        final_mu = np.mean(mus, axis=0)

        if return_std:
            aleatoric_var = np.mean(np.exp(log_vars), axis=0)
            epistemic_var = np.var(mus, axis=0)
            total_std = np.sqrt(aleatoric_var + epistemic_var)
            return final_mu, total_std

        return final_mu


# ===================================================================
# TRAINING PIPELINE
# ===================================================================

X_train_proc = fiverr_dss_pipeline.transform(X_train)
X_test_proc = fiverr_dss_pipeline.transform(X_test)

# The column transformer returns sparse CSR matrices; convert to dense arrays
X_train_proc = X_train_proc.toarray()
X_test_proc = X_test_proc.toarray()

print("Transformed train shape (dense):", X_train_proc.shape)
print("Transformed test shape (dense):", X_test_proc.shape)

# Set random seeds for reproducibility (before model creation)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)

# 2. Create and train the heteroscedastic model
hnn = HeteroscedasticKerasRegressor(
    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    verbose=1,
    random_state=RANDOM_STATE,
)

hnn.fit(X_train_proc, y_train)

# 3. Predict on the test set (with uncertainty)
y_pred_mean, y_pred_std = hnn.predict(X_test_proc, return_std=True)

# 4. Evaluation metrics: R^2, MAE, MAPE
r2 = r2_score(y_test, y_pred_mean)
mae = mean_absolute_error(y_test, y_pred_mean)
mape = mean_absolute_percentage_error(y_test, y_pred_mean)

print("\n=== Heteroscedastic NN Performance (Newton + StopGrad) ===")
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

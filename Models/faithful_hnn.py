import numpy as np
import random
from sklearn.base import BaseEstimator, RegressorMixin
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from Dataset.preprocessing import fiverr_dss_pipeline, X_train, X_test, y_train, y_test
import joblib

def aleatoric_loss(y_true, y_pred):
    mu = y_pred[:, :1]
    log_var = y_pred[:, 1:]
    return K.mean(0.5 * K.exp(-log_var) * K.square(y_true - mu) + 0.5 * log_var)

def build_hnn_model(input_dim, learning_rate=0.001, random_state=None):
    if random_state is not None:
        tf.random.set_seed(random_state)

    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x, training=True)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x, training=True)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(2)(x)  # [mean, log_var]

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=aleatoric_loss)
    return model

class HeteroscedasticKerasRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, epochs=100, batch_size=32, learning_rate=0.001, n_samples=50, verbose=0, random_state=42):
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

        es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        # ---- Stage 1: Train mean head only with MSE ----
        for layer in self.model_.layers:
            if 'logvar' in layer.name or 'var_' in layer.name:
                layer.trainable = False
        self.model_.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss='mse',
        )
        self.model_.fit(
            X, y,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=[es],
        )

        # ---- Stage 2: Freeze mean head, train variance head with NLL ----
        for layer in self.model_.layers:
            layer.trainable = True
        for layer in self.model_.layers:
            if 'mu_head' in layer.name or 'trunk' in layer.name:
                layer.trainable = False
        self.model_.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss=aleatoric_loss,
        )
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
        
        # We collect n_samples of [mean, log_var]
        mus = []
        log_vars = []
        
        for _ in range(self.n_samples):
            preds = self.model_.predict(X, verbose=0)
            mus.append(preds[:, 0])
            log_vars.append(preds[:, 1])
        
        mus = np.array(mus)           # Shape: (n_samples, n_rows)
        log_vars = np.array(log_vars) # Shape: (n_samples, n_rows)

        # Final Point Estimate: Mean of the means
        final_mu = np.mean(mus, axis=0)

        if return_std:
            # TOTAL UNCERTAINTY = Aleatoric + Epistemic
            # Aleatoric (Data noise): Average of the predicted variances
            aleatoric_var = np.mean(np.exp(log_vars), axis=0)
            
            # Epistemic (Model uncertainty): Variance of the predicted means
            epistemic_var = np.var(mus, axis=0)
            
            total_std = np.sqrt(aleatoric_var + epistemic_var)
            return final_mu, total_std
            
        return final_mu



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
    verbose=1,        # set to 0 to silence training logs
    random_state=RANDOM_STATE,
)

hnn.fit(X_train_proc, y_train)

# 3. Predict on the test set (with uncertainty)
y_pred_mean, y_pred_std = hnn.predict(X_test_proc, return_std=True)

# 4. Evaluation metrics in dollar-space (expm1 to invert log1p)
y_test_dollars = np.expm1(y_test)
y_pred_dollars = np.expm1(y_pred_mean)

r2 = r2_score(y_test_dollars, y_pred_dollars)
mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
mape = mean_absolute_percentage_error(y_test_dollars, y_pred_dollars)

print("\n=== Faithful Heteroscedastic NN Performance (dollar-space) ===")
print(f"R^2   : {r2:.4f}")
print(f"MAE   : ${mae:.2f}")
print(f"MAPE  : {mape:.4f}")

# 5. Optional: inspect a few predictions with uncertainty
for i in range(5):
    print(
        f"True: ${np.expm1(y_test.iloc[i]):.2f}, "
        f"Pred: ${np.expm1(y_pred_mean[i]):.2f}, "
        f"Std (uncertainty): {y_pred_std[i]:.3f}"
    )


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K
from sklearn.base import BaseEstimator, RegressorMixin

class FaithfulHNN(tf.keras.Model):
    def __init__(self, input_dim):
        super().__init__()
        
        # Shared trunk
        self.trunk = models.Sequential([
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
        ])
        
        # Mean head
        self.mean_head = layers.Dense(1)
        
        # Variance head (log-variance)
        self.var_head = layers.Dense(1)

    def call(self, inputs, training=False):
        z = self.trunk(inputs, training=training)
        
        # Mean uses full trunk
        mu = self.mean_head(z)
        
        # Variance head receives stop-gradient trunk
        z_stop = tf.stop_gradient(z)
        log_var = self.var_head(z_stop)
        
        return mu, log_var


def gaussian_nll(y_true, mu, log_var):
    return 0.5 * tf.exp(-log_var) * tf.square(y_true - mu) + 0.5 * log_var


class FaithfulHNNTrainer:
    def __init__(self, input_dim, learning_rate=1e-3):
        self.model = FaithfulHNN(input_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            mu, log_var = self.model(x, training=True)
            loss = tf.reduce_mean(gaussian_nll(y, mu, log_var))

        grads = tape.gradient(loss, self.model.trainable_variables)

        # Newton scaling for mean-head gradients
        sigma2 = tf.exp(log_var)

        new_grads = []
        for var, g in zip(self.model.trainable_variables, grads):
            if g is None:
                new_grads.append(None)
                continue

            # Identify mean-head variables
            if "mean_head" in var.name:
                # Multiply gradient by sigma^2 (Newton step)
                # Clip variance scaling to prevent overshooting (between 0.1 and 10.0)
                sigma2_clipped = tf.clip_by_value(sigma2, 0.1, 10.0)
                g = g * sigma2_clipped

            new_grads.append(g)

        self.optimizer.apply_gradients(zip(new_grads, self.model.trainable_variables))
        return loss

class FaithfulHeteroscedasticRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        epochs=100,
        batch_size=32,
        learning_rate=0.0005,  # Lower default LR due to Newton scaling
        verbose=0,
        validation_split=0.2,
        patience=15,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.validation_split = validation_split
        self.patience = patience
        self.trainer = None

    def fit(self, X, y):
        X = X.astype("float32")
        y = np.asarray(y, dtype="float32").reshape(-1, 1)

        # Split into train and validation
        n_train = int(len(X) * (1 - self.validation_split))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        self.trainer = FaithfulHNNTrainer(X.shape[1], self.learning_rate)

        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(2048).batch(self.batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(self.batch_size)

        # Early stopping tracking
        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None

        for epoch in range(self.epochs):
            # Training
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in train_dataset:
                loss = self.trainer.train_step(xb, yb)
                epoch_loss += loss.numpy()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches if n_batches > 0 else epoch_loss

            # Validation
            val_loss = 0.0
            val_batches = 0
            for xb, yb in val_dataset:
                mu, log_var = self.trainer.model(xb, training=False)
                loss = tf.reduce_mean(gaussian_nll(yb, mu, log_var))
                val_loss += loss.numpy()
                val_batches += 1

            avg_val_loss = val_loss / val_batches if val_batches > 0 else val_loss

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best weights
                best_weights = [w.numpy().copy() for w in self.trainer.model.trainable_variables]
            else:
                patience_counter += 1

            if self.verbose > 0 and epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
                    f"val_loss={avg_val_loss:.4f}, best_val_loss={best_val_loss:.4f}"
                )

            if patience_counter >= self.patience:
                if self.verbose > 0:
                    print(f"Early stopping at epoch {epoch}")
                break

        # Restore best weights
        if best_weights is not None:
            for var, best_w in zip(self.trainer.model.trainable_variables, best_weights):
                var.assign(best_w)

        return self

    def predict(self, X, return_std=False):
        X = X.astype("float32")
        mu, log_var = self.trainer.model(X, training=False)
        mu = mu.numpy().flatten()
        std = tf.sqrt(tf.exp(log_var)).numpy().flatten()

        return (mu, std) if return_std else mu

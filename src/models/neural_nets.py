"""
src/models/neural_nets.py
==========================
Neural network predictors NN1–NN5 from GKX (2020) Table 3.

Architecture (all variants)
----------------------------
- Input: rank-normalised characteristics (already in [−1, +1])
- Hidden layers: K × 32 ReLU units with L1 regularisation and Dropout (0.30)
- Output: 1 linear unit (no activation)
- Optimiser: Adam, lr = 0.001
- Loss: MSE
- Batch size: 10 000
- Max epochs: 100 with early stopping (patience = 5)

NN1 has 1 hidden layer, NN2 has 2, ..., NN5 has 5.

Each NN variant is trained with 10 random seeds; predictions are averaged
before returning (GKX §3 — "ensemble of 10 initialisations").
"""

from __future__ import annotations

import logging
import random

import numpy as np

log = logging.getLogger(__name__)

# Ensemble seeds (0–9) — must never be changed after training begins
NN_SEEDS: list[int] = list(range(10))

# Shared architecture defaults
_UNITS = 32
_DROPOUT = 0.30
_L1 = 1e-5
_LR = 1e-3
_BATCH = 10_000
_MAX_EPOCHS = 100
_PATIENCE = 5


# ---------------------------------------------------------------------------
# Single-seed Keras model builder
# ---------------------------------------------------------------------------

def _build_keras_model(n_features: int, n_layers: int) -> "tf.keras.Model":
    """Build a single Keras model with n_layers hidden layers."""
    import tensorflow as tf
    from tensorflow import keras

    reg = keras.regularizers.l1(_L1)
    inputs = keras.Input(shape=(n_features,), name="chars")
    x = inputs
    for i in range(n_layers):
        x = keras.layers.Dense(
            _UNITS,
            activation="relu",
            kernel_regularizer=reg,
            name=f"hidden_{i}",
        )(x)
        x = keras.layers.Dropout(_DROPOUT, name=f"drop_{i}")(x)
    outputs = keras.layers.Dense(1, name="ret_pred")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=_LR),
        loss="mse",
    )
    return model


# ---------------------------------------------------------------------------
# NNEnsemble — sklearn-compatible estimator
# ---------------------------------------------------------------------------

class NNEnsemble:
    """
    Ensemble of `len(seeds)` Keras models.  `predict()` returns the average
    of all seeds' predictions.

    Parameters
    ----------
    n_layers : int — number of hidden layers (1 for NN1, ..., 5 for NN5)
    seeds    : list[int] — random seeds for initialisation
    """

    def __init__(self, n_layers: int, seeds: list[int] = NN_SEEDS) -> None:
        self.n_layers = n_layers
        self.seeds = seeds
        self._models: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NNEnsemble":
        import tensorflow as tf
        from tensorflow import keras

        n_features = X.shape[1]
        self._models = []

        for seed in self.seeds:
            # Seed everything for reproducibility
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

            model = _build_keras_model(n_features, self.n_layers)

            cb = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=_PATIENCE,
                restore_best_weights=True,
            )

            model.fit(
                X, y,
                batch_size=_BATCH,
                epochs=_MAX_EPOCHS,
                validation_split=0.1,
                callbacks=[cb],
                verbose=0,
            )
            self._models.append(model)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError("NNEnsemble.fit() must be called before predict().")
        preds = np.stack(
            [m.predict(X, verbose=0).ravel() for m in self._models],
            axis=1,
        )
        return preds.mean(axis=1)


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def make_nn(n_layers: int, seeds: list[int] = NN_SEEDS) -> NNEnsemble:
    """Return an NNEnsemble with `n_layers` hidden layers."""
    return NNEnsemble(n_layers=n_layers, seeds=seeds)


def make_nn1(seeds: list[int] = NN_SEEDS) -> NNEnsemble:
    return make_nn(1, seeds)

def make_nn2(seeds: list[int] = NN_SEEDS) -> NNEnsemble:
    return make_nn(2, seeds)

def make_nn3(seeds: list[int] = NN_SEEDS) -> NNEnsemble:
    return make_nn(3, seeds)

def make_nn4(seeds: list[int] = NN_SEEDS) -> NNEnsemble:
    return make_nn(4, seeds)

def make_nn5(seeds: list[int] = NN_SEEDS) -> NNEnsemble:
    return make_nn(5, seeds)

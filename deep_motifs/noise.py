from __future__ import annotations

import numpy as np


def _apply_feature_noise(
    X: np.ndarray,
    noise_type: str,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_type == "none" or noise_level == 0.0:
        return X
    X = X.copy().astype(np.float32)
    if noise_type == "gaussian":
        stds = X.std(axis=0)
        X += (rng.standard_normal(X.shape) * (noise_level * stds)).astype(np.float32)
    elif noise_type == "dropout":
        X[rng.random(X.shape) < noise_level] = 0.0
    return X


def _apply_label_noise(
    y: np.ndarray,
    flip_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if flip_rate == 0.0:
        return y
    y = y.copy()
    neg_idx = np.where(y == 0)[0]
    n_flip = int(len(neg_idx) * flip_rate)
    if n_flip > 0:
        y[rng.choice(neg_idx, size=n_flip, replace=False)] = 1
    return y

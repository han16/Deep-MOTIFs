from __future__ import annotations

import numpy as np


# ============================================================
# Threshold & calibration
# ============================================================

def find_best_threshold_by_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    beta: float = 1.0,
) -> float:
    """
    Search for the threshold that maximises F-beta score.

    beta=1.0  → standard F1 (precision = recall equal weight)
    beta<1.0  → precision-biased  (beta=0.8: precision weighted ~1.6× recall)
    beta>1.0  → recall-biased

    Per-fold calibration uses beta=1.0 (default).
    Global OOF threshold uses beta<1.0 to prevent precision collapse
    when the pooled OOF distribution shifts the optimal F1 threshold down.
    """
    from sklearn.metrics import fbeta_score
    y_true  = np.asarray(y_true,  dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.5
    grid = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 99)))
    best_t, best_score = 0.5, -1.0
    for t in grid:
        score = float(fbeta_score(y_true, (y_score >= t).astype(int),
                                  beta=beta, zero_division=0))
        if score > best_score:
            best_t, best_score = float(t), score
    return float(np.clip(best_t, 1e-4, 1.0 - 1e-4))


def remap_score_with_threshold(y_score: np.ndarray, threshold: float) -> np.ndarray:
    s = np.clip(np.asarray(y_score, dtype=float), 1e-6, 1.0 - 1e-6)
    t = float(np.clip(threshold, 1e-4, 1.0 - 1e-4))
    new_odds = (s / (1.0 - s)) / (t / (1.0 - t))
    return np.clip(new_odds / (1.0 + new_odds), 1e-6, 1.0 - 1e-6)


def recall_at_k_score(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true  = np.asarray(y_true,  dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0:
        return float("nan")
    k   = int(max(min(k, y_true.size), 1))
    pos = float(np.sum(y_true == 1))
    if pos <= 0:
        return float("nan")
    return float(np.sum(y_true[np.argsort(-y_score)[:k]] == 1) / pos)
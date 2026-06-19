from __future__ import annotations

import numpy as np
import pandas as pd


def apply_training_label_perturbation(
    train_df: pd.DataFrame,
    mode: str = "none",
    rate: float = 0.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    """Apply label perturbation to a fold's training labels only.

    Modes:
    - none: leave labels unchanged.
    - negative_contamination: flip a fraction of training negatives to positives.
      This simulates weak reference negatives that are actually undiscovered ASD genes.
    - positive_hiding: flip a fraction of training positives to negatives.
      This simulates incomplete positive annotation, where known positives are hidden
      from the learner and treated as non-positive reference genes.

    The held-out test fold should be passed separately and must not be perturbed.
    """
    mode = str(mode).strip().lower()
    rate = float(rate)
    out = train_df.copy()
    y = out["label"].to_numpy(dtype=int).copy()
    rng = np.random.default_rng(int(random_state))

    info: dict[str, int | float | str] = {
        "label_noise_mode": mode,
        "label_noise_rate": rate,
        "n_flipped": 0,
        "n_train_before": int(len(out)),
        "n_pos_before": int((y == 1).sum()),
        "n_neg_before": int((y == 0).sum()),
    }

    if mode in {"none", ""} or rate <= 0.0:
        out["label"] = y
    elif mode == "negative_contamination":
        idx = np.flatnonzero(y == 0)
        n = min(len(idx), int(round(len(idx) * rate)))
        if n > 0:
            chosen = rng.choice(idx, size=n, replace=False)
            y[chosen] = 1
        info["n_flipped"] = int(n)
        out["label"] = y
    elif mode == "positive_hiding":
        idx = np.flatnonzero(y == 1)
        n = min(len(idx), int(round(len(idx) * rate)))
        if n > 0:
            chosen = rng.choice(idx, size=n, replace=False)
            y[chosen] = 0
        info["n_flipped"] = int(n)
        out["label"] = y
    else:
        raise ValueError(
            f"Unknown label_noise_mode={mode!r}; expected none, "
            "negative_contamination, or positive_hiding"
        )

    info["n_pos_after"] = int((out["label"].to_numpy(dtype=int) == 1).sum())
    info["n_neg_after"] = int((out["label"].to_numpy(dtype=int) == 0).sum())
    return out, info


def add_label_noise_args(parser):
    parser.add_argument(
        "--label-noise-mode",
        type=str,
        default="none",
        choices=["none", "negative_contamination", "positive_hiding"],
        help="Training-label perturbation mode for robustness experiments.",
    )
    parser.add_argument(
        "--label-noise-rate",
        type=float,
        default=0.0,
        help="Fraction of eligible training labels perturbed in each fold.",
    )
    return parser


def apply_training_label_budget(
    train_df: pd.DataFrame,
    positive_fraction: float = 1.0,
    neg_ratio: float = 1.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    """Subsample training labels for label-budget experiments.

    The held-out test fold must not be passed here. We keep only a fraction of
    training positives and sample training negatives to a fixed negative:positive
    ratio. This evaluates low-label training while preserving the full clean test
    fold in each cross-validation split.
    """
    positive_fraction = float(positive_fraction)
    neg_ratio = float(neg_ratio)
    out = train_df.copy().reset_index(drop=True)
    rng = np.random.default_rng(int(random_state))

    pos_idx = np.flatnonzero(out["label"].to_numpy(dtype=int) == 1)
    neg_idx = np.flatnonzero(out["label"].to_numpy(dtype=int) == 0)
    info: dict[str, int | float | str] = {
        "label_budget_positive_fraction": positive_fraction,
        "label_budget_neg_ratio": neg_ratio,
        "n_train_before_budget": int(len(out)),
        "n_pos_before_budget": int(len(pos_idx)),
        "n_neg_before_budget": int(len(neg_idx)),
    }

    if positive_fraction >= 0.999 and neg_ratio <= 0:
        info["n_train_after_budget"] = int(len(out))
        info["n_pos_after_budget"] = int(len(pos_idx))
        info["n_neg_after_budget"] = int(len(neg_idx))
        return out, info

    if positive_fraction >= 0.999 and neg_ratio > 0:
        n_pos_keep = len(pos_idx)
    else:
        positive_fraction = float(np.clip(positive_fraction, 1e-6, 1.0))
        n_pos_keep = max(1, int(round(len(pos_idx) * positive_fraction)))

    chosen_pos = (
        rng.choice(pos_idx, size=n_pos_keep, replace=False)
        if n_pos_keep < len(pos_idx)
        else pos_idx
    )

    if neg_ratio > 0:
        n_neg_keep = min(len(neg_idx), max(1, int(round(n_pos_keep * neg_ratio))))
        chosen_neg = (
            rng.choice(neg_idx, size=n_neg_keep, replace=False)
            if n_neg_keep < len(neg_idx)
            else neg_idx
        )
    else:
        chosen_neg = neg_idx

    keep = np.concatenate([chosen_pos, chosen_neg])
    keep.sort()
    out = out.iloc[keep].copy().reset_index(drop=True)

    info["n_train_after_budget"] = int(len(out))
    info["n_pos_after_budget"] = int((out["label"].to_numpy(dtype=int) == 1).sum())
    info["n_neg_after_budget"] = int((out["label"].to_numpy(dtype=int) == 0).sum())
    return out, info


def add_label_budget_args(parser):
    parser.add_argument(
        "--label-budget-positive-fraction",
        type=float,
        default=1.0,
        help="Fraction of training positives retained in each fold for label-budget experiments.",
    )
    parser.add_argument(
        "--label-budget-neg-ratio",
        type=float,
        default=0.0,
        help=(
            "Training negative:positive ratio after label-budget subsampling. "
            "Use 1.0 for balanced low-label experiments; <=0 keeps all training negatives."
        ),
    )
    return parser

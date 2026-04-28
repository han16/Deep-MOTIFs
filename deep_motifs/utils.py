from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch

from .xgb import coerce_numeric_and_impute


# ============================================================
# Noise injection utilities
# ============================================================

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


# ============================================================
# Reproducibility
# ============================================================

def set_torch_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# ============================================================
# Feature preparation
# ============================================================

def build_view_frames(
    meta_df: pd.DataFrame,
    brainspan_df: pd.DataFrame,
    string_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta_cols = meta_df.columns.tolist()
    keep_meta_cols = [c for i, c in enumerate(meta_cols, start=1) if i > 7]
    work_meta = meta_df[keep_meta_cols] if keep_meta_cols else meta_df.copy()
    meta_num = coerce_numeric_and_impute(work_meta)

    bs_median = brainspan_df.median(numeric_only=True)
    bs_all = brainspan_df.reindex(meta_df.index).fillna(bs_median)
    bs_all = bs_all.replace([np.inf, -np.inf], np.nan)
    bs_all = bs_all.fillna(bs_all.median(numeric_only=True)).fillna(0.0)
    bs_all = coerce_numeric_and_impute(bs_all)

    str_all = string_df.reindex(meta_df.index)
    str_all = str_all.replace([np.inf, -np.inf], np.nan)
    str_all = str_all.fillna(str_all.median(numeric_only=True)).fillna(0.0)
    str_all = coerce_numeric_and_impute(str_all)

    return meta_num, bs_all, str_all


def standardize_fit_and_all(x_fit: np.ndarray, x_all: np.ndarray) -> np.ndarray:
    mu = x_fit.mean(axis=0, keepdims=True)
    sigma = x_fit.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    out = (x_all - mu) / sigma
    out = np.clip(np.nan_to_num(out, nan=0.0, posinf=30.0, neginf=-30.0), -30.0, 30.0)
    return out.astype(np.float32)


def repeat_array(x: np.ndarray, factor: int) -> np.ndarray:
    factor = int(max(factor, 1))
    return x if factor <= 1 else np.tile(x, factor)
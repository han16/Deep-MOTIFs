from __future__ import annotations

import numpy as np
import pandas as pd

from .xgb import coerce_numeric_and_impute


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

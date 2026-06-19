from __future__ import annotations

import numpy as np
import pandas as pd

from xgb import coerce_numeric_and_impute


def compute_meta_top_pairs(
    meta_df: "pd.DataFrame",
    labels_df: "pd.DataFrame",
    top_k: int = 6,
    random_state: int = 42,
) -> "tuple[list[tuple[str, str]], list[str]]":
    """
    Train a quick XGBoost on composite_table meta features (columns i>7, same as
    build_view_frames) to identify the top-K most important features by gain.

    Returns:
      top_pairs   — all C(top_k, 2) column-name pairs for pairwise products
      top_squares — top_k column names for squared features
    """
    from xgboost import XGBClassifier

    meta_cols = meta_df.columns.tolist()
    keep_cols = [c for i, c in enumerate(meta_cols, start=1) if i > 7]
    meta_num  = coerce_numeric_and_impute(
        meta_df[keep_cols] if keep_cols else meta_df.copy()
    )

    label_ids = labels_df["id"].tolist()
    y         = labels_df["label"].to_numpy(dtype=int)
    X_labeled = meta_num.reindex(label_ids).fillna(0.0)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    spw   = float(n_neg / max(n_pos, 1))

    clf = XGBClassifier(
        n_estimators=100, max_depth=4, min_child_weight=5,
        reg_alpha=0.1, gamma=0.1, scale_pos_weight=spw,
        random_state=random_state, n_jobs=-1,
        objective="binary:logistic", tree_method="hist",
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    )
    clf.fit(X_labeled.to_numpy(dtype=np.float32), y)

    importance  = pd.Series(clf.feature_importances_, index=X_labeled.columns)
    top_k       = min(top_k, len(importance))
    top_feats   = importance.nlargest(top_k).index.tolist()

    top_pairs   = [(top_feats[i], top_feats[j])
                   for i in range(len(top_feats))
                   for j in range(i + 1, len(top_feats))]
    top_squares = top_feats
    return top_pairs, top_squares


def poly_expand_meta(
    meta_df: "pd.DataFrame",
    top_pairs:   "list[tuple[str, str]]",
    top_squares: "list[str]",
) -> "pd.DataFrame":
    """
    Append polynomial interaction features to the meta DataFrame.

    For each (col_a, col_b) in top_pairs  → adds col_a × col_b
    For each col in top_squares            → adds col²

    Note: meta_df values are already coerce_numeric_and_impute'd (finite floats).
    Downstream standardise_fit_and_all normalises everything together.
    """
    extra: dict[str, np.ndarray] = {}
    for col_a, col_b in top_pairs:
        if col_a in meta_df.columns and col_b in meta_df.columns:
            extra[f"poly_{col_a}_x_{col_b}"] = (
                meta_df[col_a].to_numpy(dtype=np.float32)
                * meta_df[col_b].to_numpy(dtype=np.float32)
            )
    for col in top_squares:
        if col in meta_df.columns:
            extra[f"poly_{col}_sq"] = meta_df[col].to_numpy(dtype=np.float32) ** 2

    if not extra:
        return meta_df
    poly_df = pd.DataFrame(extra, index=meta_df.index, dtype=np.float32)
    return pd.concat([meta_df, poly_df], axis=1)

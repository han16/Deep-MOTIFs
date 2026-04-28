from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .xgb import coerce_numeric_and_impute


# ============================================================
# v25: XGB-guided polynomial meta-feature expansion
# ============================================================

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


# ============================================================
# XGBoost OOF warm-start feature
# ============================================================

def _build_xgb_feature_matrix(
    meta_all: pd.DataFrame,
    bs_all: pd.DataFrame,
    str_all: pd.DataFrame,
) -> pd.DataFrame:
    X = pd.concat(
        [meta_all.astype(np.float32),
         bs_all.astype(np.float32),
         str_all.astype(np.float32)],
        axis=1,
    )
    X.columns = [str(c) for c in X.columns]
    return X.loc[:, ~X.columns.duplicated()]


def _fit_xgb_v18(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_all: pd.DataFrame,
    n_estimators: int,
    random_state: int,
    max_depth: int = 4,
    min_child_weight: int = 5,
    reg_alpha: float = 0.1,
    gamma: float = 0.1,
) -> tuple:
    """
    v18 optimised XGBoost fit.  Compared to the default fit_xgb_and_score:
      max_depth        6  鈫?4    (shallower trees, less overfitting on ~665 labels)
      min_child_weight 1  鈫?5    (require more samples per leaf)
      reg_alpha        0  鈫?0.1  (L1 regularisation for feature sparsity)
      gamma            0  鈫?0.1  (minimum gain required to make a split)
    """
    from xgboost import XGBClassifier
    from sklearn.dummy import DummyClassifier

    y_unique = np.unique(y_train)
    if y_unique.size < 2:
        constant = int(y_unique[0]) if y_unique.size == 1 else 0
        clf = DummyClassifier(strategy="constant", constant=constant)
        clf.fit(X_train, y_train)
        scores = np.full(X_all.shape[0], float(constant), dtype=float)
        return clf, pd.Series(scores, index=X_all.index)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        learning_rate=0.05,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        reg_lambda=1.0,
        reg_alpha=reg_alpha,
        gamma=gamma,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_all)
    pos_col = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
    return clf, pd.Series(proba[:, pos_col], index=X_all.index, dtype=float)


def compute_xgb_oof_scores(
    labels_df: pd.DataFrame,
    meta_all: pd.DataFrame,
    bs_all: pd.DataFrame,
    str_all: pd.DataFrame,
    n_splits: int,
    random_state: int,
    n_estimators: int = 500,
    xgb_max_depth: int = 4,
    xgb_min_child_weight: int = 5,
    xgb_reg_alpha: float = 0.1,
    xgb_gamma: float = 0.1,
    cache_path: Path | None = None,
) -> pd.Series:
    """Out-of-fold XGBoost scores 鈥?zero leakage. v18: optimised hyperparameters."""
    if cache_path is not None and cache_path.exists():
        print(f"[INFO] Loading XGBoost OOF scores from cache: {cache_path}")
        return pd.read_csv(cache_path, index_col=0).iloc[:, 0]

    X_all = _build_xgb_feature_matrix(meta_all, bs_all, str_all)
    label_ids = labels_df["id"].tolist()
    y_all     = labels_df["label"].to_numpy(dtype=int)
    oof_scores = pd.Series(0.5, index=X_all.index, dtype=float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + 999)
    for fold_i, (inner_tr, inner_val) in enumerate(skf.split(label_ids, y_all), start=1):
        tr_ids  = [label_ids[i] for i in inner_tr]
        val_ids = [label_ids[i] for i in inner_val]
        _, fold_scores = _fit_xgb_v18(
            X_train=X_all.loc[tr_ids],
            y_train=y_all[inner_tr],
            X_all=X_all.loc[val_ids],
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=xgb_max_depth,
            min_child_weight=xgb_min_child_weight,
            reg_alpha=xgb_reg_alpha,
            gamma=xgb_gamma,
        )
        oof_scores.loc[val_ids] = fold_scores.values
        print(f"  [XGB-OOF] inner fold {fold_i}/{n_splits} done")

    label_id_set   = set(label_ids)
    unlabelled_ids = [i for i in X_all.index if i not in label_id_set]
    if unlabelled_ids:
        _, full_scores = _fit_xgb_v18(
            X_train=X_all.loc[label_ids],
            y_train=y_all,
            X_all=X_all.loc[unlabelled_ids],
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=xgb_max_depth,
            min_child_weight=xgb_min_child_weight,
            reg_alpha=xgb_reg_alpha,
            gamma=xgb_gamma,
        )
        oof_scores.loc[unlabelled_ids] = full_scores.values

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        oof_scores.to_csv(cache_path, header=["xgb_oof_score"])
        print(f"[INFO] XGBoost OOF scores cached to: {cache_path}")

    return oof_scores
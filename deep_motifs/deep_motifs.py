from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import WeightedRandomSampler

from xgb import build_brainspan_matrix
from xgb import build_fold_string_feature_matrix
from xgb import build_string_graph
from xgb import coerce_numeric_and_impute
from xgb import compute_graph_features
from xgb import ensure_exists
from xgb import evaluate_predictions
from xgb import fit_xgb_and_score
from xgb import augment_composite_with_tada
from xgb import load_composite_table
from xgb import load_labels


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


# ============================================================
# 鏂瑰悜涓夛細鍒嗘暟铻嶅悎宸ュ叿鍑芥暟
# ============================================================

def search_optimal_alpha(
    y_true: np.ndarray,
    xgb_scores: np.ndarray,
    pu_scores: np.ndarray,
    n_grid: int = 21,
) -> tuple[float, float]:
    """
    鍦ㄩ獙璇侀泦涓婄綉鏍兼悳绱㈡渶浼樿瀺鍚堟潈閲?伪锛?        fused = 伪 脳 xgb_score + (1-伪) 脳 pu_score
    杩斿洖 (best_alpha, best_pr_auc)

    伪=1.0 鈫?绾?XGBoost
    伪=0.0 鈫?绾?Deep-MOTIFs
    伪=0.5 鈫?鍚勫崐铻嶅悎
    """
    y_true     = np.asarray(y_true,     dtype=int)
    xgb_scores = np.asarray(xgb_scores, dtype=float)
    pu_scores  = np.asarray(pu_scores,  dtype=float)

    if np.unique(y_true).size < 2:
        return 0.5, float("nan")

    best_alpha, best_auc = 0.5, -1.0
    for alpha in np.linspace(0.0, 1.0, n_grid):
        fused = alpha * xgb_scores + (1.0 - alpha) * pu_scores
        auc   = float(average_precision_score(y_true, fused))
        if auc > best_auc:
            best_alpha, best_auc = float(alpha), auc
    return best_alpha, best_auc


def fuse_scores(
    xgb_scores: np.ndarray,
    pu_scores: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """鐢ㄧ粰瀹?alpha 铻嶅悎涓ょ粍鍒嗘暟銆?"""
    fused = float(alpha) * np.asarray(xgb_scores, dtype=float) \
          + (1.0 - float(alpha)) * np.asarray(pu_scores, dtype=float)
    return np.clip(fused, 1e-6, 1.0 - 1e-6)


def rrf_fuse_scores(
    xgb_scores: np.ndarray,
    pu_scores: np.ndarray,
    k: int = 60,
) -> np.ndarray:
    """
    Reciprocal Rank Fusion (Cormack et al., 2009).

    RRF_score(d) = 1/(k + rank_xgb(d)) + 1/(k + rank_pu(d))

    rank is 1-based (rank 1 = highest score).
    k controls the balance between top-rank dominance and uniform contribution:
      small k 鈫?top items dominate; large k 鈫?ranks contribute more uniformly.
      Classic default: k=60.

    Output is min-max normalised to [1e-6, 1-1e-6] so it stays
    compatible with the downstream threshold calibration.
    """
    xgb_scores = np.asarray(xgb_scores, dtype=float)
    pu_scores  = np.asarray(pu_scores,  dtype=float)
    n = len(xgb_scores)

    xgb_rank = np.empty(n, dtype=np.float64)
    pu_rank  = np.empty(n, dtype=np.float64)
    xgb_rank[np.argsort(-xgb_scores)] = np.arange(1, n + 1)
    pu_rank [np.argsort(-pu_scores )] = np.arange(1, n + 1)

    rrf = 1.0 / (k + xgb_rank) + 1.0 / (k + pu_rank)

    rrf_min, rrf_max = rrf.min(), rrf.max()
    if rrf_max > rrf_min:
        rrf = (rrf - rrf_min) / (rrf_max - rrf_min)
    return np.clip(rrf, 1e-6, 1.0 - 1e-6).astype(np.float32)


def asymmetric_rrf_fuse(
    arr1: np.ndarray,
    arr2: np.ndarray,
    k: int = 60,
    ppr_w: float = 0.7,
) -> np.ndarray:
    """
    v20: Asymmetric Reciprocal Rank Fusion.

    score = 1/(k + rank_arr1) + ppr_w/(k + rank_arr2)

    ppr_w=1.0  鈫?standard symmetric RRF (same as rrf_fuse_scores)
    ppr_w=0.7  鈫?arr1 contributes ~59%, arr2 contributes ~41%
    ppr_w=0.5  鈫?arr1 contributes ~67%, arr2 contributes ~33%

    Crucially, this PRESERVES the right-skewed RRF output distribution,
    so threshold calibration remains stable (unlike rank-normalised fusion
    which collapses the distribution to uniform and breaks calibration).

    Default ppr_w=0.7: model gets ~59% effective weight, PPR gets ~41%.
    Output is min-max normalised to [1e-6, 1-1e-6].
    """
    arr1 = np.asarray(arr1, dtype=float)
    arr2 = np.asarray(arr2, dtype=float)
    n = len(arr1)

    rank1 = np.empty(n, dtype=np.float64)
    rank2 = np.empty(n, dtype=np.float64)
    rank1[np.argsort(-arr1)] = np.arange(1, n + 1)
    rank2[np.argsort(-arr2)] = np.arange(1, n + 1)

    rrf = 1.0 / (k + rank1) + ppr_w / (k + rank2)

    rrf_min, rrf_max = rrf.min(), rrf.max()
    if rrf_max > rrf_min:
        rrf = (rrf - rrf_min) / (rrf_max - rrf_min)
    return np.clip(rrf, 1e-6, 1.0 - 1e-6).astype(np.float32)


# ============================================================
# v16: Personalized PageRank score propagation on STRING graph
# ============================================================

def propagate_scores_ppr(
    scores: pd.Series,
    G,
    alpha: float = 0.5,
    n_iter: int = 30,
    min_edge_weight: float = 0.5,
) -> pd.Series:
    """
    Personalized PageRank (Random Walk with Restart) on the STRING network.

    r_{t+1} = alpha * p0 + (1 - alpha) * A_norm @ r_t

    alpha         : restart probability. Higher 鈫?stays closer to original scores.
                    alpha=1.0 means no propagation (identity).
    min_edge_weight: filter out STRING edges below this normalised weight before
                    propagation. Default 0.5 corresponds to STRING score > 700
                    when the graph was built with base threshold=400
                    (weight = (score-400)/600, so 0.5 鈫?score > 700).

    Genes not present in the STRING graph keep their original score unchanged.
    Output is min-max normalised to [1e-6, 1-1e-6].
    """
    import scipy.sparse as sp

    # Restrict to genes present in both scores index and graph
    nodes = [n for n in scores.index if n in G]
    if not nodes:
        return scores.copy()

    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Build sparse weighted adjacency (symmetric, filtered by weight)
    rows, cols, data = [], [], []
    for u, v, d in G.edges(nbunch=nodes, data=True):
        if u not in node_idx or v not in node_idx:
            continue
        w = d.get("weight", 1.0)
        if w < min_edge_weight:
            continue
        i, j = node_idx[u], node_idx[v]
        rows += [i, j]
        cols += [j, i]
        data += [w, w]

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)

    # Row-normalise 鈫?stochastic transition matrix
    row_sums = np.asarray(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    A_norm = sp.diags(1.0 / row_sums) @ A

    # Personalisation vector: original scores (non-negative, normalised to sum=1)
    p0 = scores.reindex(nodes).fillna(0.0).to_numpy(dtype=np.float64)
    p0 = np.clip(p0, 0.0, None)
    p0_sum = p0.sum()
    if p0_sum > 0:
        p0 = p0 / p0_sum
    else:
        p0 = np.ones(n, dtype=np.float64) / n

    # Power iteration
    r = p0.copy()
    for _ in range(n_iter):
        r_new = alpha * p0 + (1.0 - alpha) * A_norm.dot(r)
        if np.max(np.abs(r_new - r)) < 1e-8:
            r = r_new
            break
        r = r_new

    # Min-max normalise to [1e-6, 1-1e-6]
    r_min, r_max = r.min(), r.max()
    if r_max > r_min:
        r = (r - r_min) / (r_max - r_min)
    r = np.clip(r, 1e-6, 1.0 - 1e-6)

    # Write back; genes absent from graph keep original score
    result = scores.copy().astype(float)
    for node, i in node_idx.items():
        result[node] = float(r[i])
    return result


def compute_ppr_from_seeds(
    seed_ids: list,
    all_ids: list,
    G,
    alpha: float = 0.5,
    n_iter: int = 30,
    min_edge_weight: float = 0.5,
    seed_weights: dict | None = None,
) -> pd.Series:
    """
    v17: Seeded Personalized PageRank.

    p鈧€ = uniform distribution over seed_ids (known positive genes).
    All other genes start at 0.

    v20: seed_weights (optional dict {gene_id: weight}) allows confidence-weighted
    seeds. If provided, each seed's initial probability is proportional to its
    weight (e.g. XGBoost OOF score). High-confidence positives contribute more
    to the propagation signal, reducing noise from uncertain training positives.

    r_{t+1} = alpha * p0 + (1 - alpha) * A_norm @ r_t

    Returns a pd.Series indexed by all_ids representing each gene's
    network proximity to the seed (known ASD) genes.
    Genes absent from the STRING graph receive score 0.0.
    Output is min-max normalised to [1e-6, 1-1e-6].
    """
    import scipy.sparse as sp

    seed_set = set(seed_ids)
    nodes = [n for n in all_ids if n in G]
    if not nodes or not seed_set:
        return pd.Series(0.0, index=all_ids)

    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Build sparse weighted adjacency (symmetric, filtered by weight)
    rows, cols, data = [], [], []
    for u, v, d in G.edges(nbunch=nodes, data=True):
        if u not in node_idx or v not in node_idx:
            continue
        w = d.get("weight", 1.0)
        if w < min_edge_weight:
            continue
        i, j = node_idx[u], node_idx[v]
        rows += [i, j]
        cols += [j, i]
        data += [w, w]

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)

    # Row-normalise 鈫?stochastic transition matrix
    row_sums = np.asarray(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    A_norm = sp.diags(1.0 / row_sums) @ A

    # Personalisation vector: uniform (v17) or confidence-weighted (v20) seeds
    seeds_in_graph = [s for s in seed_ids if s in node_idx]
    p0 = np.zeros(n, dtype=np.float64)
    if seeds_in_graph:
        if seed_weights is not None:
            for s in seeds_in_graph:
                p0[node_idx[s]] = seed_weights.get(s, 1.0)
        else:
            for s in seeds_in_graph:
                p0[node_idx[s]] = 1.0
        p0 /= p0.sum()
    else:
        p0 = np.ones(n, dtype=np.float64) / n

    # Power iteration
    r = p0.copy()
    for _ in range(n_iter):
        r_new = alpha * p0 + (1.0 - alpha) * A_norm.dot(r)
        if np.max(np.abs(r_new - r)) < 1e-8:
            r = r_new
            break
        r = r_new

    # Min-max normalise to [1e-6, 1-1e-6]
    r_min, r_max = r.min(), r.max()
    if r_max > r_min:
        r = (r - r_min) / (r_max - r_min)
    r = np.clip(r, 1e-6, 1.0 - 1e-6)

    # Build full result indexed by all_ids; genes not in graph get 0.0
    result = pd.Series(0.0, index=all_ids, dtype=float)
    for node, i in node_idx.items():
        result[node] = float(r[i])
    return result


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


# ============================================================
# Augmentation
# ============================================================

def make_corrupted_view(
    x: torch.Tensor, mask_rate: float, noise_std: float
) -> torch.Tensor:
    out = x.clone()
    if mask_rate > 0:
        out = out.masked_fill(torch.rand_like(out) < mask_rate, 0.0)
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    return out


# ============================================================
# Loss functions
# ============================================================

def nnpu_loss(
    pos_logits: torch.Tensor,
    unlabeled_logits: torch.Tensor,
    class_prior: float,
) -> torch.Tensor:
    if pos_logits.numel() == 0:
        return unlabeled_logits.new_tensor(0.0)
    class_prior = float(np.clip(class_prior, 1e-4, 0.95))
    pos_risk = class_prior * F.binary_cross_entropy_with_logits(
        pos_logits, torch.ones_like(pos_logits)
    )
    if unlabeled_logits.numel() == 0:
        return pos_risk
    neg_risk = (
        F.binary_cross_entropy_with_logits(
            unlabeled_logits, torch.zeros_like(unlabeled_logits)
        )
        - class_prior * F.binary_cross_entropy_with_logits(
            pos_logits, torch.zeros_like(pos_logits)
        )
    )
    return pos_risk + torch.clamp(neg_risk, min=0.0)


def pairwise_ranking_loss(
    pos_logits: torch.Tensor,
    unlabeled_logits: torch.Tensor,
) -> torch.Tensor:
    if pos_logits.numel() == 0 or unlabeled_logits.numel() == 0:
        t = pos_logits if pos_logits.numel() > 0 else unlabeled_logits
        return t.new_tensor(0.0)
    diff = pos_logits.unsqueeze(1) - unlabeled_logits.unsqueeze(0)
    return F.softplus(-diff).mean()


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


# ============================================================
# 鏀瑰姩涓€锛氬甫鏉冮噸鐨?STRING 鍥?+ 2灞傚姞鏉?GCN 棰勮仛鍚?# ============================================================

def build_weighted_string_graph(
    data_dir: Path,
    score_threshold: int = 400,
    cache_path: Path | None = None,
) -> "nx.Graph":
    """
    閲嶆柊璇诲彇 STRING 鍘熷鏂囦欢锛屽湪杈逛笂瀛樺偍褰掍竴鍖栨潈閲嶃€?    鏉冮噸 = (score - threshold) / (1000 - threshold)锛岃寖鍥?(0, 1]銆?
    xgb.py 鐨?build_string_graph 涓㈠純浜嗘潈閲嶏紝杩欓噷鐙珛璇诲彇浠ヤ繚鐣欐潈閲嶄俊鎭紝
    渚涘姞鏉?GCN 浣跨敤銆備娇鐢ㄧ嫭绔嬬殑缂撳瓨鏂囦欢锛屼笉褰卞搷 xgb.py 鐨勭紦瀛樸€?    """
    import networkx as nx
    import pickle

    if cache_path is not None and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    string_path = data_dir / "9606.protein.links.v10.txt.gz"
    if not string_path.exists():
        raise FileNotFoundError(f"STRING file not found: {string_path}")

    G = nx.Graph()
    with gzip.open(string_path, "rt") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split()
            a, b, score = parts[0], parts[1], int(parts[2])
            if score <= score_threshold:
                continue
            a = a.replace("9606.", "")
            b = b.replace("9606.", "")
            w = float(score - score_threshold) / float(1000 - score_threshold)
            G.add_edge(a, b, weight=w)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(G, f)
        print(f"[INFO] Weighted STRING graph cached to: {cache_path}")

    return G


def _gcn_aggregate_string(
    x_str: np.ndarray,
    G,
    ids_all: list[str],
    allowed_ids: set[str],
    n_layers: int = 2,
    self_weight: float = 0.5,
) -> np.ndarray:
    """
    n_layers 灞傚姞鏉?GCN 棰勮仛鍚堬細
        x_agg[i] = self_weight 脳 x[i]
                 + (1 - self_weight) 脳 危(w_ij 脳 x[j]) / 危(w_ij)

    杈规潈閲?w_ij 鏉ヨ嚜 STRING score 褰掍竴鍖栧€笺€?    鑻ュ浘鏃犳潈閲嶅睘鎬э紙fallback 鍒版棤鏉冮噸鍥撅級锛屽垯绛夋潈閲嶈仛鍚堛€?    鍙湪 allowed_ids锛坱rain universe锛夊唴鍋氶偦灞呰仛鍚堬紝閬垮厤 test 淇℃伅娉勬紡銆?    """
    lookup  = {pid: i for i, pid in enumerate(ids_all)}
    allowed = set(allowed_ids)
    x       = x_str.copy().astype(np.float32)

    for _layer in range(n_layers):
        x_new = x.copy()
        for pid, i in lookup.items():
            if pid not in G or pid not in allowed:
                continue
            # 鏀堕泦甯︽潈閲嶇殑閭诲眳
            nbs: list[tuple[int, float]] = []
            for n in G.neighbors(pid):
                if n not in lookup or n not in allowed:
                    continue
                # 浼樺厛浣跨敤杈规潈閲嶏紝鏃犳潈閲嶆椂榛樿 1.0
                w = float(G[pid][n].get("weight", 1.0))
                nbs.append((lookup[n], w))
            if not nbs:
                continue
            total_w = sum(w for _, w in nbs)
            if total_w <= 0:
                continue
            nb_agg   = sum(w * x[j] for j, w in nbs) / total_w
            x_new[i] = self_weight * x[i] + (1.0 - self_weight) * nb_agg
        x = x_new

    return x.astype(np.float32)


# ============================================================
# FIX 1: MetaMLP 鈥?single MLP projection, no group splitting
# Preserves cross-feature correlations that GroupedTokenizer destroys.
# ============================================================

class MetaMLP(nn.Module):
    """
    Projects the full meta feature vector through a 2-layer MLP into a
    single token of size token_dim.  Unlike GroupedTokenizer, this sees
    all features simultaneously, preserving pairwise interactions 鈥?the
    same interactions that make XGBoost strong on composite_table data.
    """
    def __init__(self, in_dim: int, token_dim: int, dropout: float) -> None:
        super().__init__()
        hidden = max(token_dim * 2, 64)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns [B, 1, token_dim] 鈥?one token representing all meta features
        return self.net(x).unsqueeze(1)


# ============================================================
# Grouped tokenizer (kept for STRING view)
# ============================================================

def build_group_sizes(total_dim: int, n_groups: int) -> list[int]:
    total_dim = int(max(total_dim, 1))
    n_groups  = int(max(min(n_groups, total_dim), 1))
    base, rem = divmod(total_dim, n_groups)
    out = [base + (1 if i < rem else 0) for i in range(n_groups)]
    return [v for v in out if v > 0]


class GroupedTokenizer(nn.Module):
    def __init__(self, in_dim: int, n_groups: int, token_dim: int, dropout: float):
        super().__init__()
        self.group_sizes = build_group_sizes(in_dim, n_groups)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(g, token_dim),
                nn.LayerNorm(token_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for g in self.group_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(x, self.group_sizes, dim=1)
        return torch.stack(
            [proj(c) for proj, c in zip(self.projections, chunks)], dim=1
        )


# ============================================================
# BrainSpan temporal encoder
# ============================================================

class BrainSpanEncoder(nn.Module):
    """
    v5 upgrade: cross-region attention added after per-region temporal conv.

    Pipeline:
      1. Reshape (B, 16*50) 鈫?(B*16, 1, 50)
      2. Per-region temporal Conv1d 鈫?pool 鈫?(B, 16, token_dim)
         Captures *within-region* developmental trajectories.
      3. Cross-region TransformerEncoderLayer over the 16 region tokens
         Captures *between-region* coordinated expression patterns 鈥?a signal
         that XGBoost cannot model because it treats all 800 BrainSpan columns
         as independent flat features.

    Output: (B, 16, token_dim)  鈥?same shape as before, drop-in replacement.
    """
    def __init__(
        self,
        total_bs_dim: int,
        n_regions: int,
        n_timepoints: int,
        token_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.n_regions    = n_regions
        self.n_timepoints = n_timepoints
        self.token_dim    = token_dim
        self.structured   = (total_bs_dim == n_regions * n_timepoints)

        if self.structured:
            # Step 1-2: per-region temporal encoding (unchanged from v4)
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(1, token_dim // 2, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(token_dim // 2, token_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.norm = nn.LayerNorm(token_dim)

            # Step 3: cross-region attention (new in v5)
            # n_heads must divide token_dim; use 4 for token_dim=64
            _n_heads = max(token_dim // 16, 1)
            self.region_attn = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=_n_heads,
                dim_feedforward=token_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,   # pre-norm: more stable on short sequences
            )
            self.drop = nn.Dropout(dropout)
        else:
            self.fallback = GroupedTokenizer(total_bs_dim, n_regions, token_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.structured:
            return self.fallback(x)
        B = x.shape[0]
        # Per-region temporal encoding
        x = x.contiguous().view(B * self.n_regions, 1, self.n_timepoints)
        h = self.temporal_conv(x)
        h = self.pool(h).squeeze(-1).view(B, self.n_regions, self.token_dim)
        h = self.norm(h)
        # Cross-region attention: lets regions communicate
        h = self.region_attn(h)   # (B, 16, token_dim)
        return self.drop(h)


# ============================================================
# DeepMOTIFs (Deep Multi-Omics Transformer with Integrated Features and Scores)
# ============================================================

class DeepMOTIFs(nn.Module):
    """
    Changes vs previous version:
      FIX 1: meta_tok replaced by MetaMLP (single MLP, no group splitting)
      FIX 6: norm_first=False (restores PyTorch nested-tensor optimisation)
    """

    def __init__(
        self,
        meta_dim: int,
        bs_dim: int,
        str_dim: int,
        token_dim: int,
        bs_n_regions: int,
        bs_n_timepoints: int,
        str_token_count: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.token_dim = int(token_dim)

        # FIX 1: single MLP for meta (was GroupedTokenizer with 4 groups)
        self.meta_tok = MetaMLP(meta_dim, self.token_dim, dropout)
        self.bs_tok   = BrainSpanEncoder(
            bs_dim, bs_n_regions, bs_n_timepoints, self.token_dim, dropout
        )
        self.str_tok  = GroupedTokenizer(str_dim, str_token_count, self.token_dim, dropout)

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.type_embed = nn.Parameter(torch.zeros(1, 4, self.token_dim))

        # FIX 6: norm_first=False 鈥?restores nested-tensor speed optimisation
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=int(max(n_heads, 1)),
            dim_feedforward=int(max(self.token_dim * 4, 64)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(max(n_layers, 1)))
        self.norm     = nn.LayerNorm(self.token_dim)

        bottleneck = max(64, self.token_dim)
        self.shared_bottleneck = nn.Sequential(
            nn.Linear(self.token_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pu_head   = nn.Linear(bottleneck, 1)
        self.rank_head = nn.Linear(bottleneck, 1)

        nn.init.normal_(self.cls_token,  std=0.02)
        nn.init.normal_(self.type_embed, std=0.02)

    def forward(
        self, x_m: torch.Tensor, x_b: torch.Tensor, x_s: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        B  = x_m.shape[0]
        tm = self.meta_tok(x_m)   # [B, 1, D]
        tb = self.bs_tok(x_b)     # [B, n_regions, D]
        ts = self.str_tok(x_s)    # [B, str_token_count, D]
        cls = self.cls_token.expand(B, -1, -1)

        seq = torch.cat([cls, tm, tb, ts], dim=1)
        type_seq = torch.cat([
            self.type_embed[:, 0:1, :].expand(B, 1,           self.token_dim),
            self.type_embed[:, 1:2, :].expand(B, tm.shape[1], self.token_dim),
            self.type_embed[:, 2:3, :].expand(B, tb.shape[1], self.token_dim),
            self.type_embed[:, 3:4, :].expand(B, ts.shape[1], self.token_dim),
        ], dim=1)
        seq = seq + type_seq

        emb    = self.norm(self.encoder(seq)[:, 0, :])
        shared = self.shared_bottleneck(emb)
        return {
            "emb":        emb,
            "pu_logit":   self.pu_head(shared).squeeze(1),
            "rank_logit": self.rank_head(shared).squeeze(1),
        }


# ============================================================
# Vectorised neighbour matrix
# ============================================================

def build_neighbor_matrix(
    G,
    ids: list[str],
    allowed_ids: set[str],
    top_k: int = 3,
) -> np.ndarray:
    lookup  = {pid: i for i, pid in enumerate(ids)}
    allowed = set(allowed_ids)
    matrix  = np.full((len(ids), top_k), -1, dtype=np.int64)
    for pid, i in lookup.items():
        if pid not in G or pid not in allowed:
            continue
        nbs = [
            (n, G.degree(n))
            for n in G.neighbors(pid)
            if n in lookup and n in allowed
        ]
        if not nbs:
            continue
        nbs.sort(key=lambda x: -x[1])
        for k_idx, (n, _) in enumerate(nbs[:top_k]):
            matrix[i, k_idx] = lookup[n]
    return matrix


# ============================================================
# OC centre 鈥?epoch-level
# ============================================================

def compute_pos_center(
    model: DeepMOTIFs,
    meta_t: torch.Tensor,
    bs_t: torch.Tensor,
    str_t: torch.Tensor,
    pos_global_idx: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    embs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(pos_global_idx), batch_size):
            idx = torch.from_numpy(
                pos_global_idx[start:start + batch_size].astype(np.int64)
            ).long()
            embs.append(
                model(meta_t[idx].to(device),
                      bs_t[idx].to(device),
                      str_t[idx].to(device))["emb"]
            )
    center = torch.cat(embs, 0).mean(0).detach()
    model.train()
    return center


# ============================================================
# v6: Masked Feature Reconstruction Pre-training
# ============================================================

def pretrain_encoder(
    model: DeepMOTIFs,
    meta_t: torch.Tensor,
    bs_t: torch.Tensor,
    str_dim: int,
    device: torch.device,
    pretrain_epochs: int,
    pretrain_lr: float,
    pretrain_mask_rate: float,
    batch_size: int,
    pos_global_idx: np.ndarray,   # global indices of positive samples in the full gene array
    w_pretrain_pu: float = 0.3,   # weight for nnPU in pretrain loss
    progress_prefix: str = "",
) -> None:
    """
    Self-supervised pre-training on ALL genes (no labels needed).

    Pipeline per step:
      1. Sample a random batch of genes.
      2. Randomly mask pretrain_mask_rate of meta and BrainSpan features (set to 0).
      3. Pass zero STRING features (fold-independent 鈥?avoids anchor leakage).
         str_tok is deliberately excluded from the pretrain optimizer so its weights
         stay at random init and are not biased toward zero-input patterns.
      4. Decode CLS token via two lightweight linear heads 鈫?reconstructed meta & bs.
      5. MSE loss only on masked positions.

    This lets the encoder see all 16,609 genes before fold-specific fine-tuning,
    leveraging unlabeled data in a way XGBoost cannot.
    """
    meta_dim  = meta_t.shape[1]
    bs_dim    = bs_t.shape[1]
    token_dim = model.token_dim

    meta_decoder = nn.Linear(token_dim, meta_dim).to(device)
    bs_decoder   = nn.Linear(token_dim, bs_dim).to(device)

    # Exclude str_tok: its input is all-zeros during pretrain, so updating it
    # would bias its weights toward zero-input patterns, hurting fine-tuning.
    str_tok_ids  = {id(p) for p in model.str_tok.parameters()}
    pretrain_model_params = [p for p in model.parameters() if id(p) not in str_tok_ids]
    all_pretrain_params   = (
        pretrain_model_params
        + list(meta_decoder.parameters())
        + list(bs_decoder.parameters())
    )

    pretrain_opt = torch.optim.AdamW(all_pretrain_params, lr=pretrain_lr, weight_decay=1e-4)

    n_genes = meta_t.shape[0]
    ds      = TensorDataset(torch.arange(n_genes, dtype=torch.long))
    loader  = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model.train()
    meta_decoder.train()
    bs_decoder.train()

    pos_set = set(pos_global_idx.tolist())
    class_prior = float(np.clip(len(pos_set) / meta_t.shape[0], 1e-4, 0.95))

    for epoch_idx in range(pretrain_epochs):
        epoch_losses: list[float] = []
        for (idx_b,) in loader:
            idx_b = idx_b.long()
            B     = idx_b.shape[0]

            xm = meta_t[idx_b].to(device)                          # (B, meta_dim)
            xb = bs_t[idx_b].to(device)                            # (B, bs_dim)
            xs = torch.zeros(B, str_dim, device=device)            # (B, str_dim) 鈥?zeros

            # Random masks
            mask_m = torch.rand(B, meta_dim, device=device) < pretrain_mask_rate
            mask_b = torch.rand(B, bs_dim,   device=device) < pretrain_mask_rate

            xm_masked = xm.masked_fill(mask_m, 0.0)
            xb_masked = xb.masked_fill(mask_b, 0.0)

            out   = model(xm_masked, xb_masked, xs)
            emb   = out["emb"]                                      # (B, token_dim)

            rec_m = meta_decoder(emb)                               # (B, meta_dim)
            rec_b = bs_decoder(emb)                                 # (B, bs_dim)

            # Loss only on masked positions
            loss_m = F.mse_loss(rec_m[mask_m], xm[mask_m]) if mask_m.any() else rec_m.new_tensor(0.0)
            loss_b = F.mse_loss(rec_b[mask_b], xb[mask_b]) if mask_b.any() else rec_b.new_tensor(0.0)
            loss   = loss_m + loss_b

            # nnPU loss during pretrain: pos vs unlabeled
            pos_in_batch = torch.tensor(
                [j for j, gi in enumerate(idx_b.tolist()) if gi in pos_set],
                dtype=torch.long, device=device,
            )
            unl_in_batch = torch.tensor(
                [j for j, gi in enumerate(idx_b.tolist()) if gi not in pos_set],
                dtype=torch.long, device=device,
            )
            if pos_in_batch.numel() > 0 and unl_in_batch.numel() > 0:
                l_pu_pre = nnpu_loss(
                    out["pu_logit"][pos_in_batch],
                    out["pu_logit"][unl_in_batch],
                    class_prior,
                )
                loss = loss + w_pretrain_pu * l_pu_pre

            if not torch.isfinite(loss):
                continue

            pretrain_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_pretrain_params, 1.0)
            pretrain_opt.step()
            epoch_losses.append(float(loss.item()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        if (epoch_idx + 1) % 10 == 0 or epoch_idx == 0:
            print(
                f"{progress_prefix}[Pretrain] Epoch {epoch_idx+1}/{pretrain_epochs} "
                f"loss={mean_loss:.6f}"
            )

    model.train()  # leave in train mode for subsequent fine-tuning
    return meta_decoder, bs_decoder


# ============================================================
# v25: Pretrain reconstruction-error feature importance
# ============================================================

def compute_pretrain_meta_importance(
    model: "DeepMOTIFs",
    meta_decoder: "nn.Linear",
    meta_t: torch.Tensor,
    bs_t: torch.Tensor,
    str_dim: int,
    device: torch.device,
    meta_col_names: "list[str]",
    top_k: int = 6,
    batch_size: int = 512,
) -> "tuple[list[tuple[str, str]], list[str]]":
    """
    Rank meta features by masked-reconstruction MSE using the pretrained encoder.

    For each feature j, set column j to 0 across all genes, run a forward pass,
    and measure MSE of the decoder prediction for column j vs the true value.
    Higher MSE → harder to predict from other features → more unique information.
    No labels are used: zero label leakage.
    """
    model.eval()
    meta_decoder.eval()

    n_genes  = meta_t.shape[0]
    meta_dim = meta_t.shape[1]

    meta_t_dev = meta_t.to(device)
    bs_t_dev   = bs_t.to(device)
    xs_zeros   = torch.zeros(n_genes, str_dim, device=device)

    feature_mse: list[float] = []
    with torch.no_grad():
        for j in range(meta_dim):
            xm_masked = meta_t_dev.clone()
            xm_masked[:, j] = 0.0  # ablate feature j only

            rec_parts: list[torch.Tensor] = []
            for start in range(0, n_genes, batch_size):
                end = min(start + batch_size, n_genes)
                out = model(
                    xm_masked[start:end],
                    bs_t_dev[start:end],
                    xs_zeros[start:end],
                )
                rec = meta_decoder(out["emb"])[:, j].cpu()
                rec_parts.append(rec)

            rec_j  = torch.cat(rec_parts)
            true_j = meta_t_dev[:, j].cpu()
            feature_mse.append(float(F.mse_loss(rec_j, true_j).item()))

    model.train()
    meta_decoder.train()

    importance = pd.Series(feature_mse, index=meta_col_names)
    top_k      = min(top_k, len(importance))
    top_feats  = importance.nlargest(top_k).index.tolist()

    top_pairs   = [(top_feats[i], top_feats[j])
                   for i in range(len(top_feats))
                   for j in range(i + 1, len(top_feats))]
    top_squares = top_feats
    return top_pairs, top_squares


# ============================================================
# DataLoader helper
# ============================================================

def cycle_next(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


# ============================================================
# Main training function
# ============================================================

def fit_deep_motifs_and_export(
    X_meta_all_raw: pd.DataFrame,
    X_bs_all_raw: pd.DataFrame,
    X_str_all_raw: pd.DataFrame,
    ids_all: list[str],
    train_df: pd.DataFrame,
    test_ids: set[str],
    G,
    random_state: int,
    device: torch.device,
    token_dim: int,
    bs_n_regions: int,
    bs_n_timepoints: int,
    str_token_count: int,
    transformer_heads: int,
    transformer_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    early_stop_metric: str,
    early_stop_k: int,
    augment_factor: int,
    augment_scale: float,
    mask_rate_meta: float,
    mask_rate_bs: float,
    mask_rate_str: float,
    noise_std: float,
    w_bce: float,
    w_pu: float,
    w_rank: float,
    w_oc: float,
    w_graph: float,
    w_cons: float,
    graph_top_k: int,
    xgb_oof_scores: pd.Series | None,
    weighted_G,
    gcn_n_layers: int = 2,
    warmup_epochs: int = 10,
    center_update_interval: int = 5,
    use_torch_compile: bool = False,
    pretrain_epochs: int = 50,
    pretrain_lr: float = 1e-3,
    pretrain_mask_rate: float = 0.30,
    w_pretrain_pu: float = 0.3,
    ckpt_avg_k: int = 5,
    progress_prefix: str = "",
) -> tuple[pd.Series, pd.DataFrame, dict[str, float]]:

    set_torch_seed(random_state)
    all_lookup = {pid: i for i, pid in enumerate(ids_all)}

    train_ids     = train_df["id"].tolist()
    y_train       = train_df["label"].to_numpy(dtype=int)
    train_pos_ids = [p for p, y in zip(train_ids, y_train.tolist()) if int(y) == 1]

    universe_ids = [pid for pid in ids_all if pid not in test_ids]
    if not universe_ids:
        raise ValueError("No train universe ids left after excluding test fold ids.")
    fit_idx = np.asarray([all_lookup[p] for p in universe_ids], dtype=np.int64)

    # v2 FIX 2: Do NOT append XGB OOF score as a model input feature.
    # In pu.py, the XGB OOF score was appended to the meta view, making Deep-MOTIFs
    # condition its representation on XGBoost's prediction.  This prevents the model
    # from learning signals orthogonal to XGB, and makes the post-hoc fusion redundant.
    # XGB OOF scores are still used for post-fusion (outside this function), but the
    # neural network must learn from raw features only so the two models remain complementary.
    X_meta_final = X_meta_all_raw

    # Standardise (fit on train universe only)
    x_meta = standardize_fit_and_all(
        X_meta_final.to_numpy(dtype=np.float32)[fit_idx],
        X_meta_final.to_numpy(dtype=np.float32),
    )
    x_bs = standardize_fit_and_all(
        X_bs_all_raw.to_numpy(dtype=np.float32)[fit_idx],
        X_bs_all_raw.to_numpy(dtype=np.float32),
    )
    x_str = standardize_fit_and_all(
        X_str_all_raw.to_numpy(dtype=np.float32)[fit_idx],
        X_str_all_raw.to_numpy(dtype=np.float32),
    )

    # ---- 鏂瑰悜浜岋細GCN 棰勮仛鍚?STRING 鐗瑰緛 ----
    # 鐢?STRING 鍥剧殑褰掍竴鍖栭偦鎺ョ煩闃靛 x_str 鍋氫竴娆″浘鑱氬悎锛?    #   x_str_agg[i] = x_str[i] + mean(x_str[neighbours of i])
    # ---- 鏀瑰姩涓€锛氬姞鏉冨灞?GCN 棰勮仛鍚?STRING 鐗瑰緛 ----
    x_str = _gcn_aggregate_string(
        x_str=x_str,
        G=weighted_G,
        ids_all=ids_all,
        allowed_ids=set(universe_ids),
        n_layers=gcn_n_layers,
        self_weight=0.5,
    )

    # Build model
    model = DeepMOTIFs(
        meta_dim=x_meta.shape[1],
        bs_dim=x_bs.shape[1],
        str_dim=x_str.shape[1],
        token_dim=token_dim,
        bs_n_regions=bs_n_regions,
        bs_n_timepoints=bs_n_timepoints,
        str_token_count=str_token_count,
        n_heads=transformer_heads,
        n_layers=transformer_layers,
        dropout=dropout,
    ).to(device)

    if use_torch_compile:
        try:
            model = torch.compile(model)
            print(f"{progress_prefix}[torch.compile] OK")
        except Exception as e:
            print(f"{progress_prefix}[torch.compile] Skipped: {e}")

    meta_t = torch.from_numpy(x_meta).float()
    bs_t   = torch.from_numpy(x_bs).float()
    str_t  = torch.from_numpy(x_str).float()

    # Index arrays (train_pos_idx needed before pretrain for nnPU pretrain loss)
    train_global_idx = np.asarray([all_lookup[p] for p in train_ids],     dtype=np.int64)
    train_pos_idx    = np.asarray([all_lookup[p] for p in train_pos_ids], dtype=np.int64)

    # v6: Masked Feature Reconstruction Pre-training on all genes (no labels).
    # Runs before fold-specific fine-tuning so the encoder has seen the full gene universe.
    if pretrain_epochs > 0:
        print(
            f"{progress_prefix}[Pretrain] Starting masked pre-training "
            f"({pretrain_epochs} epochs, lr={pretrain_lr}, mask_rate={pretrain_mask_rate})"
        )
        pretrain_encoder(
            model=model,
            meta_t=meta_t,
            bs_t=bs_t,
            str_dim=x_str.shape[1],
            device=device,
            pretrain_epochs=pretrain_epochs,
            pretrain_lr=pretrain_lr,
            pretrain_mask_rate=pretrain_mask_rate,
            batch_size=batch_size,
            pos_global_idx=train_pos_idx,
            w_pretrain_pu=w_pretrain_pu,
            progress_prefix=progress_prefix,
        )
        print(f"{progress_prefix}[Pretrain] Done. Starting fine-tuning...")

    universe_idx     = np.asarray([all_lookup[p] for p in universe_ids],  dtype=np.int64)
    pos_set          = set(train_pos_idx.tolist())
    # v2 FIX 1: unlabeled_idx only contains truly unlabeled genes (not in any labeled set).
    # In pu.py, labeled negatives (label=0) were included in unlabeled_idx, causing BCE and
    # nnPU to apply contradictory gradients to the same samples.  Here we exclude all
    # labeled genes (pos AND neg) from the nnPU unlabeled pool, so the two losses operate
    # on disjoint sets: BCE handles supervised labels, nnPU handles truly unlabeled genes.
    labeled_train_set = set(all_lookup[pid] for pid in train_ids)  # pos + neg
    unlabeled_idx     = np.asarray(
        [i for i in universe_idx.tolist() if i not in labeled_train_set], dtype=np.int64
    )
    if unlabeled_idx.size == 0:
        # fallback: if no truly-unlabeled genes exist, revert to pos-excluded set
        unlabeled_idx = np.asarray(
            [i for i in universe_idx.tolist() if i not in pos_set], dtype=np.int64
        )
    if unlabeled_idx.size == 0:
        unlabeled_idx = universe_idx.copy()

    # Train / val split
    idx = np.arange(len(train_ids))
    stratify_arg = (
        y_train
        if len(np.unique(y_train)) > 1 and min(np.bincount(y_train)) >= 2
        else None
    )
    tr_idx, val_idx = train_test_split(
        idx, test_size=0.25, random_state=random_state,
        shuffle=True, stratify=stratify_arg,
    )
    y_tr  = y_train[tr_idx]
    y_val = y_train[val_idx]
    tr_global  = train_global_idx[tr_idx]
    val_global = train_global_idx[val_idx]

    tr_pos_global = np.asarray(
        [g for g, yv in zip(tr_global.tolist(), y_tr.tolist()) if int(yv) == 1],
        dtype=np.int64,
    )
    if tr_pos_global.size == 0:
        tr_pos_global = train_pos_idx.copy()

    # Augmented arrays
    tr_global_aug = repeat_array(tr_global,     augment_factor)
    y_tr_aug      = repeat_array(y_tr,          augment_factor)
    tr_pos_aug    = repeat_array(tr_pos_global, augment_factor)
    unlabeled_aug = repeat_array(unlabeled_idx, augment_factor)

    # DataLoaders
    labeled_ds   = TensorDataset(torch.from_numpy(tr_global_aug).long(),
                                  torch.from_numpy(y_tr_aug.astype(np.float32)).float())
    pos_ds       = TensorDataset(torch.from_numpy(tr_pos_aug).long())
    unlabeled_ds = TensorDataset(torch.from_numpy(unlabeled_aug).long())
    val_ds       = TensorDataset(torch.from_numpy(val_global).long(),
                                  torch.from_numpy(y_val.astype(np.float32)).float())

    class_counts   = np.bincount(y_tr_aug.astype(int), minlength=2)
    class_counts[class_counts == 0] = 1
    sample_weights = np.clip((1.0 / class_counts)[y_tr_aug.astype(int)], 1e-6, None)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    pin = device.type == "cuda"
    labeled_loader   = DataLoader(labeled_ds,   batch_size=batch_size, sampler=sampler,
                                   num_workers=0, pin_memory=pin)
    pos_loader       = DataLoader(pos_ds,        batch_size=batch_size, shuffle=True,
                                   num_workers=0, pin_memory=pin)
    unlabeled_loader = DataLoader(unlabeled_ds,  batch_size=batch_size, shuffle=True,
                                   num_workers=0, pin_memory=pin)
    val_loader       = DataLoader(val_ds,        batch_size=batch_size, shuffle=False,
                                   num_workers=0, pin_memory=pin)

    nb_matrix = build_neighbor_matrix(
        G, ids_all, allowed_ids=set(universe_ids), top_k=graph_top_k
    )

    # Loss setup
    n_pos_tr = int((y_tr == 1).sum())
    n_neg_tr = int((y_tr == 0).sum())
    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(n_neg_tr / max(n_pos_tr, 1)),
                                dtype=torch.float32, device=device)
    )
    class_prior = float(np.clip(len(train_pos_ids) / max(len(universe_ids), 1), 1e-3, 0.5))

    # Optimiser + Warmup 鈫?Cosine
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    _warmup       = max(int(warmup_epochs), 1)
    _cosine_steps = max(epochs - _warmup, 1)
    warmup_sched  = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=_warmup
    )
    cosine_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=_cosine_steps, eta_min=learning_rate * 0.05
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched], milestones=[_warmup]
    )
    print(f"{progress_prefix}Scheduler: {_warmup} warmup 鈫?cosine {_cosine_steps} epochs")

    # Early stopping + v7 Checkpoint Averaging
    # ckpt_buffer stores the last ckpt_avg_k improved checkpoints.
    # After training, their weights are averaged (Polyak averaging) to smooth
    # out the noise in selecting the single best epoch 鈥?reduces PR-AUC variance.
    maximize    = early_stop_metric in {"pr_auc", "recall_at_k"}
    best_metric = -float("inf") if maximize else float("inf")
    best_state  = copy.deepcopy(model.state_dict())
    ckpt_buffer: deque = deque(maxlen=max(int(ckpt_avg_k), 1))
    bad_epochs  = 0
    pos_center: torch.Tensor | None = None

    # Pre-compute augmentation constants
    aug_mm  = float(augment_scale * mask_rate_meta)
    aug_mb  = float(augment_scale * mask_rate_bs)
    aug_ms  = float(augment_scale * mask_rate_str)
    aug_n   = float(augment_scale * noise_std)
    cons_mm = float(mask_rate_meta * 0.5)
    cons_mb = float(mask_rate_bs   * 0.5)
    cons_ms = float(mask_rate_str  * 0.5)
    cons_n  = float(noise_std      * 0.5)

    def gather(idx: torch.Tensor):
        return (meta_t[idx].to(device),
                bs_t[idx].to(device),
                str_t[idx].to(device))

    # ============================================================
    # Training loop
    # ============================================================
    # Self-paced PU 鈥?姣忎釜 epoch 閮芥洿鏂帮紝褰诲簳閬垮厤杩囨湡鍒嗘暟闂
    sp_threshold     = 0.7
    sp_threshold_min = 0.3
    sp_update_every  = 1      # 姣?epoch 鏇存柊锛屼唬浠峰緢灏忥紙unlabeled 鍏ㄩ噺鎺ㄧ悊锛?    reliable_neg_mask_u: np.ndarray | None = None

    for epoch_idx in range(epochs):

        if pos_center is None or epoch_idx % center_update_interval == 0:
            pos_center = compute_pos_center(
                model, meta_t, bs_t, str_t, tr_pos_global, device, batch_size
            )

        # ---- 鏂瑰悜涓€锛歋elf-paced PU 鈥?鍔ㄦ€佹洿鏂板彲闈犺礋鏍锋湰 mask ----
        # 姣?sp_update_every 涓?epoch锛岀敤褰撳墠妯″瀷瀵?unlabeled 鍩哄洜鎵撳垎锛?        # 鍙繚鐣欏垎鏁颁綆浜?sp_threshold 鐨勫熀鍥犲弬涓?unlabeled loss锛?        # 鎺掗櫎閭ｄ簺妯″瀷璁や负"鍙兘鏄鏍锋湰"鐨勫熀鍥犮€?        # sp_threshold 闅忚缁冭繘琛岀嚎鎬ц“鍑忥紙瓒婃潵瓒婂鏉撅級銆?        if epoch_idx % sp_update_every == 0:
            model.eval()
            with torch.no_grad():
                u_scores_list: list[np.ndarray] = []
                for start in range(0, len(unlabeled_idx), batch_size):
                    end   = min(start + batch_size, len(unlabeled_idx))
                    idx_u_sp = torch.from_numpy(
                        unlabeled_idx[start:end].astype(np.int64)
                    ).long()
                    sp_out = model(*gather(idx_u_sp))
                    u_scores_list.append(
                        torch.sigmoid(sp_out["pu_logit"]).cpu().numpy()
                    )
            u_scores_all = np.concatenate(u_scores_list)
            reliable_neg_mask_u = u_scores_all < sp_threshold

            # 姣?5 涓?epoch 鎵嶈“鍑忎竴娆?threshold锛堝叡琛板噺 epochs/5 娆★級
            if epoch_idx % 5 == 0:
                decay_per_update = (0.7 - sp_threshold_min) / max(epochs / 5, 1)
                sp_threshold = max(sp_threshold - decay_per_update, sp_threshold_min)

            n_reliable = int(reliable_neg_mask_u.sum())
            # 姣?5 涓?epoch 鎵撳嵃涓€娆★紝閬垮厤鏃ュ織杩囧
            if epoch_idx % 5 == 0:
                print(
                    f"{progress_prefix}[SP-PU] epoch={epoch_idx+1} "
                    f"reliable_neg={n_reliable}/{len(unlabeled_idx)} "
                    f"({100*n_reliable/max(len(unlabeled_idx),1):.1f}%) "
                    f"threshold={sp_threshold:.3f}"
                )
            model.train()

        # v2 FIX 3: Dynamic label propagation removed from training loop.
        # In pu.py, pseudo-positive genes were injected every 10 epochs, which
        # created a feedback loop with SP-PU (both depend on u_scores_all) that
        # destabilised training.  Label propagation, if needed at all, should be
        # done once after the model has converged 鈥?not interleaved with gradient updates.

        # 鐢?reliable mask 绛涢€夋湰 epoch 鐨?unlabeled 鏍锋湰
        if reliable_neg_mask_u is not None and reliable_neg_mask_u.sum() > 0:
            reliable_u_idx = unlabeled_aug[
                np.tile(reliable_neg_mask_u, augment_factor)[:len(unlabeled_aug)]
            ]
            if reliable_u_idx.size == 0:
                reliable_u_idx = unlabeled_aug
        else:
            reliable_u_idx = unlabeled_aug

        reliable_u_ds     = TensorDataset(torch.from_numpy(reliable_u_idx).long())
        reliable_u_loader = DataLoader(
            reliable_u_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=pin,
        )
        it_u = iter(reliable_u_loader)

        model.train()
        losses: list[float] = []
        it_l = iter(labeled_loader)
        it_p = iter(pos_loader)
        # it_u is built above by self-paced logic
        n_steps = len(labeled_loader)

        for _ in range(n_steps):
            (idx_l, y_l), it_l = cycle_next(it_l, labeled_loader)
            (idx_p,),     it_p = cycle_next(it_p, pos_loader)
            (idx_u,),     it_u = cycle_next(it_u, reliable_u_loader)
            idx_l = idx_l.long()
            idx_p = idx_p.long()
            idx_u = idx_u.long()
            y_l   = y_l.to(device).clamp(0.0, 1.0)

            xm_l, xb_l, xs_l = gather(idx_l)
            xm_p, xb_p, xs_p = gather(idx_p)
            xm_u, xb_u, xs_u = gather(idx_u)

            xm_l = make_corrupted_view(xm_l, aug_mm, aug_n)
            xb_l = make_corrupted_view(xb_l, aug_mb, aug_n)
            xs_l = make_corrupted_view(xs_l, aug_ms, aug_n)
            xm_p = make_corrupted_view(xm_p, aug_mm, aug_n)
            xb_p = make_corrupted_view(xb_p, aug_mb, aug_n)
            xs_p = make_corrupted_view(xs_p, aug_ms, aug_n)
            xm_u = make_corrupted_view(xm_u, aug_mm, aug_n)
            xb_u = make_corrupted_view(xb_u, aug_mb, aug_n)
            xs_u = make_corrupted_view(xs_u, aug_ms, aug_n)

            out_l = model(xm_l, xb_l, xs_l)
            out_p = model(xm_p, xb_p, xs_p)
            out_u = model(xm_u, xb_u, xs_u)

            l_bce  = bce(out_l["pu_logit"], y_l)
            l_pu   = nnpu_loss(out_p["pu_logit"], out_u["pu_logit"], class_prior)
            l_rank = pairwise_ranking_loss(out_p["rank_logit"], out_u["rank_logit"])
            l_oc   = torch.mean((out_p["emb"] - pos_center).pow(2))

            xm_u2  = make_corrupted_view(xm_u, cons_mm, cons_n)
            xb_u2  = make_corrupted_view(xb_u, cons_mb, cons_n)
            xs_u2  = make_corrupted_view(xs_u, cons_ms, cons_n)
            out_u2 = model(xm_u2, xb_u2, xs_u2)
            l_cons = torch.mean(
                (torch.sigmoid(out_u["pu_logit"])
                 - torch.sigmoid(out_u2["pu_logit"])).pow(2)
            )

            idx_u_np   = idx_u.cpu().numpy()
            nb_rows    = nb_matrix[idx_u_np]
            valid_mask = nb_rows >= 0
            if valid_mask.any():
                batch_pos  = np.where(valid_mask)[0]
                nb_globals = nb_rows[valid_mask]
                anc_t  = torch.from_numpy(batch_pos).long().to(device)
                nb_t   = torch.from_numpy(nb_globals).long()
                out_nb = model(*gather(nb_t))
                l_graph = F.mse_loss(out_u["emb"][anc_t], out_nb["emb"].detach())
            else:
                l_graph = out_u["emb"].new_tensor(0.0)

            loss = (
                max(w_pu,   0.0) * l_pu
                + max(w_rank, 0.0) * l_rank
                + max(w_oc,   0.0) * l_oc
                + max(w_graph,0.0) * l_graph
                + max(w_cons, 0.0) * l_cons
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss in pu.py")

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        # Validation
        model.eval()
        val_losses: list[float] = []
        val_sc: list[np.ndarray] = []
        val_gt: list[np.ndarray] = []
        with torch.no_grad():
            for idx_v, y_v in val_loader:
                xm_v, xb_v, xs_v = gather(idx_v.long())
                y_v   = y_v.to(device).clamp(0.0, 1.0)
                out_v = model(xm_v, xb_v, xs_v)
                val_losses.append(float(bce(out_v["pu_logit"], y_v).item()))
                val_sc.append(torch.sigmoid(out_v["pu_logit"]).cpu().numpy())
                val_gt.append(y_v.cpu().numpy())

        mean_train_loss = float(np.mean(losses))    if losses    else float("nan")
        mean_val_loss   = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_sc:
            p_val    = np.concatenate(val_sc)
            y_val_np = np.concatenate(val_gt).astype(int)
            val_pr_auc   = (float(average_precision_score(y_val_np, p_val))
                            if np.unique(y_val_np).size == 2 else float("nan"))
            val_recall_k = recall_at_k_score(y_val_np, p_val, early_stop_k)
        else:
            val_pr_auc = val_recall_k = float("nan")

        current_metric = (
            val_pr_auc   if early_stop_metric == "pr_auc"      else
            val_recall_k if early_stop_metric == "recall_at_k"  else
            mean_val_loss
        )
        improved = (current_metric > best_metric) if maximize else (current_metric < best_metric)
        if improved:
            best_metric = current_metric
            best_state  = copy.deepcopy(model.state_dict())
            ckpt_buffer.append(copy.deepcopy(model.state_dict()))
            bad_epochs  = 0
        else:
            bad_epochs += 1

        print(
            f"{progress_prefix}[Finetune] Epoch {epoch_idx+1}/{epochs} "
            f"train_loss={mean_train_loss:.6f} val_loss={mean_val_loss:.6f} "
            f"val_pr_auc={val_pr_auc:.6f} "
            f"val_recall@{early_stop_k}={val_recall_k:.6f} "
            f"best={best_metric:.6f} bad={bad_epochs}/{patience}"
        )
        scheduler.step()
        if bad_epochs >= patience:
            print(f"{progress_prefix}[Finetune] Early stop at epoch {epoch_idx+1}")
            break

    # v7: Checkpoint Averaging 鈥?average the last ckpt_avg_k improved checkpoints.
    # Reduces sensitivity to the exact epoch chosen by early stopping.
    if len(ckpt_buffer) > 1:
        avg_state: dict = {}
        for key in ckpt_buffer[0].keys():
            avg_state[key] = torch.stack(
                [s[key].float() for s in ckpt_buffer]
            ).mean(dim=0).to(ckpt_buffer[0][key].dtype)
        model.load_state_dict(avg_state)
        print(f"{progress_prefix}[CkptAvg] Averaged {len(ckpt_buffer)} checkpoints.")
    else:
        model.load_state_dict(best_state)
    model.eval()

    # Score all genes
    all_emb:  list[np.ndarray] = []
    all_pu:   list[np.ndarray] = []
    all_rank: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(ids_all), batch_size):
            end   = min(start + batch_size, len(ids_all))
            idx_b = torch.arange(start, end, dtype=torch.long)
            out   = model(*gather(idx_b))
            all_emb.append(out["emb"].cpu().numpy())
            all_pu.append(torch.sigmoid(out["pu_logit"]).cpu().numpy())
            all_rank.append(torch.sigmoid(out["rank_logit"]).cpu().numpy())

    emb        = np.concatenate(all_emb)
    pu_score   = np.concatenate(all_pu)
    rank_score = np.concatenate(all_rank)

    # OC score
    if len(train_pos_ids) > 0:
        pos_idx_all = np.asarray([all_lookup[p] for p in train_pos_ids], dtype=np.int64)
        center_np   = emb[pos_idx_all].mean(axis=0, keepdims=True)
    else:
        center_np = emb.mean(axis=0, keepdims=True)
    oc_dist  = np.sqrt(np.sum((emb - center_np) ** 2, axis=1))
    scale    = float(np.std(oc_dist)) if np.std(oc_dist) > 1e-6 else 1.0
    oc_score = np.clip(np.exp(-oc_dist / scale), 0.0, 1.0)

    # FIX 2: use raw pu_score directly as final score.
    final_raw = np.clip(pu_score, 1e-6, 1.0 - 1e-6)

    threshold = find_best_threshold_by_f1(y_train, final_raw[train_global_idx])
    final_cal = remap_score_with_threshold(final_raw, threshold)

    feat = {f"emb_{i+1}": emb[:, i] for i in range(emb.shape[1])}
    feat.update({
        "pu_score": pu_score, "rank_score": rank_score, "oc_score": oc_score,
        "final_raw_score": final_raw, "final_score": final_cal,
    })
    feat_df      = pd.DataFrame(feat, index=ids_all)
    # score_series 鐢?final_raw锛堟湭缁?threshold remap锛夛紝渚涘灞傝瀺鍚堟悳绱娇鐢?    # 澶栧眰浼氱敤 train_df 瀵瑰簲鐨?val set 鎼滅储鏈€浼?alpha锛屽啀缁熶竴 remap
    score_series = pd.Series(final_raw, index=ids_all, dtype=float)
    info = {
        "best_metric":       float(best_metric),
        "threshold":         float(threshold),
        "n_universe_train":  int(len(universe_ids)),
        "n_pos_train":       int(len(train_pos_ids)),
        "n_unlabeled_train": int(len(unlabeled_idx)),
        # train ids and corresponding PU scores for outer-fold fusion search
        "train_ids":         train_ids,
        "train_pu_scores":   final_raw[train_global_idx].tolist(),
        "train_labels":      y_train.tolist(),
    }
    return score_series, feat_df, info


# ============================================================
# Cross-validation driver
# ============================================================

def run_pu(
    labels_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    brainspan_df: pd.DataFrame,
    G,
    data_dir: Path,
    output_dir: Path,
    string_mode: str,
    max_string_anchors: int,
    n_splits: int,
    random_state: int,
    force_rebuild_graph_features: bool,
    device: torch.device,
    token_dim: int,
    bs_n_regions: int,
    bs_n_timepoints: int,
    str_token_count: int,
    transformer_heads: int,
    transformer_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    early_stop_metric: str,
    early_stop_k: int,
    augment_factor: int,
    augment_scale: float,
    mask_rate_meta: float,
    mask_rate_bs: float,
    mask_rate_str: float,
    noise_std: float,
    w_bce: float,
    w_pu: float,
    w_rank: float,
    w_oc: float,
    w_graph: float,
    w_cons: float,
    graph_top_k: int,
    use_xgb_feature: bool,
    xgb_n_estimators: int,
    gcn_n_layers: int,
    warmup_epochs: int,
    center_update_interval: int,
    use_torch_compile: bool,
    force_rebuild_xgb_oof: bool,
    xgb_max_depth: int = 4,
    xgb_min_child_weight: int = 5,
    xgb_reg_alpha: float = 0.1,
    xgb_gamma: float = 0.1,
    pretrain_epochs: int = 50,
    pretrain_lr: float = 1e-3,
    pretrain_mask_rate: float = 0.30,
    w_pretrain_pu: float = 0.3,
    ckpt_avg_k: int = 5,
    fusion_mode: str = "fixed",
    rrf_k: int = 60,
    ppr_alpha: float = 1.0,
    ppr_n_iter: int = 30,
    ppr_min_edge_weight: float = 0.5,
    ppr_fusion_weight: float = 0.7,
    poly_top_k: int = 6,          # v25: top-K meta features for polynomial expansion
    ablate_string: bool = False,
    ablate_brainspan: bool = False,
    noise_type: str = "none",
    noise_level: float = 0.0,
    label_flip_rate: float = 0.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_df = labels_df.reset_index(drop=True)

    target_ids = meta_df.index.astype(str).tolist()
    valid_ids  = set(meta_df.index) & set(G.nodes)
    labels_df  = labels_df[labels_df["id"].isin(valid_ids)].reset_index(drop=True)

    n_pos = int((labels_df["label"] == 1).sum())
    n_neg = int((labels_df["label"] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Only one class after filtering. n_pos={n_pos}, n_neg={n_neg}.")
    labels_df.to_csv(output_dir / "all_labels_used.csv", index=False)

    # 鏋勫缓甯︽潈閲嶇殑 STRING 鍥撅紝鐢ㄤ簬鍔犳潈 GCN 棰勮仛鍚?    print("[INFO] Building weighted STRING graph for GCN aggregation...")
    weighted_G = build_weighted_string_graph(
        data_dir=data_dir,
        score_threshold=400,
        cache_path=output_dir.parent / "cache" / "string_graph_weighted.pkl",
    )
    print(
        f"[INFO] Weighted STRING graph: "
        f"{weighted_G.number_of_nodes()} nodes, {weighted_G.number_of_edges()} edges"
    )

    # STRING graph features (graph mode)
    string_graph_features: pd.DataFrame | None = None
    if string_mode == "graph":
        cache_path = output_dir.parent / "cache" / "string_graph_features.pkl"
        string_graph_features = compute_graph_features(
            G=G, target_ids=target_ids,
            cache_path=cache_path, force_rebuild=force_rebuild_graph_features,
        )
        string_graph_features = (
            string_graph_features.reindex(meta_df.index)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(string_graph_features.median(numeric_only=True))
            .fillna(0.0)
        )

    rng = np.random.default_rng(random_state)

    # Pre-compute XGBoost OOF scores
    xgb_oof_scores: pd.Series | None = None
    if use_xgb_feature:
        cache_dir = output_dir.parent / "cache"
        label_count = int(len(labels_df))
        pos_count = int(labels_df["label"].sum())
        xgb_extra_tag = ""
        if "xgb_max_depth" in locals():
            reg_alpha_tag = str(xgb_reg_alpha).replace(".", "p")
            gamma_tag = str(xgb_gamma).replace(".", "p")
            xgb_extra_tag = (
                f"_d{xgb_max_depth}_mcw{xgb_min_child_weight}"
                f"_ra{reg_alpha_tag}_g{gamma_tag}"
            )
        xgb_cache_tag = (
            f"{Path(__file__).stem}_{Path(output_dir).name}_"
            f"s{n_splits}_rs{random_state}_"
            f"n{label_count}_p{pos_count}_est{xgb_n_estimators}{xgb_extra_tag}"
        )
        xgb_cache_path = cache_dir / f"xgb_oof_scores_{xgb_cache_tag}.csv"
        if force_rebuild_xgb_oof and xgb_cache_path.exists():
            xgb_cache_path.unlink()
            print("[INFO] XGBoost OOF cache deleted 鈥?will recompute.")

        print("[INFO] Pre-computing XGBoost OOF scores (no leakage)...")
        oof_string_df = build_fold_string_feature_matrix(
            G=G, target_ids=target_ids,
            anchor_ids=labels_df["id"].tolist(),
            max_anchors=max_string_anchors,
        )
        oof_meta, oof_bs, oof_str = build_view_frames(
            meta_df=meta_df, brainspan_df=brainspan_df, string_df=oof_string_df,
        )
        if ablate_string:
            oof_str = pd.DataFrame(
                np.zeros(oof_str.shape, dtype=np.float32),
                index=oof_str.index, columns=oof_str.columns,
            )
        if ablate_brainspan:
            oof_bs = pd.DataFrame(
                np.zeros(oof_bs.shape, dtype=np.float32),
                index=oof_bs.index, columns=oof_bs.columns,
            )
        xgb_oof_scores = compute_xgb_oof_scores(
            labels_df=labels_df,
            meta_all=oof_meta, bs_all=oof_bs, str_all=oof_str,
            n_splits=n_splits, random_state=random_state,
            n_estimators=xgb_n_estimators,
            xgb_max_depth=xgb_max_depth,
            xgb_min_child_weight=xgb_min_child_weight,
            xgb_reg_alpha=xgb_reg_alpha,
            xgb_gamma=xgb_gamma,
            cache_path=xgb_cache_path,
        )
        pos_ids_set = set(labels_df[labels_df["label"] == 1]["id"])
        neg_ids_set = set(labels_df[labels_df["label"] == 0]["id"])
        print(
            f"[INFO] XGBoost OOF done. "
            f"pos_mean={xgb_oof_scores.loc[list(pos_ids_set)].mean():.4f}  "
            f"neg_mean={xgb_oof_scores.loc[list(neg_ids_set)].mean():.4f}"
        )
        xgb_oof_scores.to_csv(output_dir / "xgb_oof_scores.csv", header=["xgb_oof_score"])

    # v25: Pretrain reconstruction-error polynomial meta-feature expansion.
    # A temporary Deep-MOTIFs is pretrained (fully unsupervised) on ALL genes using
    # masked-feature reconstruction.  Per-feature ablation MSE ranks features with
    # ZERO label leakage — no labels are touched during importance computation.
    # The same top-K pairs/squares are used for polynomial expansion in every fold.
    top_meta_pairs:   list = []
    top_meta_squares: list = []
    if use_xgb_feature and poly_top_k > 0:
        print(
            f"[INFO] v25: pretrain-MSE feature selection "
            f"(top-{poly_top_k}, zero label leakage)..."
        )
        # Reuse oof_meta / oof_bs already built above; standardise on all genes.
        _x_meta_pt = standardize_fit_and_all(
            oof_meta.to_numpy(dtype=np.float32),
            oof_meta.to_numpy(dtype=np.float32),
        )
        _x_bs_pt = standardize_fit_and_all(
            oof_bs.to_numpy(dtype=np.float32),
            oof_bs.to_numpy(dtype=np.float32),
        )
        _str_dim_pt = oof_str.shape[1]
        _meta_t_pt  = torch.from_numpy(_x_meta_pt).float()
        _bs_t_pt    = torch.from_numpy(_x_bs_pt).float()

        _tmp_model = DeepMOTIFs(
            meta_dim=_x_meta_pt.shape[1],
            bs_dim=_x_bs_pt.shape[1],
            str_dim=_str_dim_pt,
            token_dim=token_dim,
            bs_n_regions=bs_n_regions,
            bs_n_timepoints=bs_n_timepoints,
            str_token_count=str_token_count,
            n_heads=transformer_heads,
            n_layers=transformer_layers,
            dropout=dropout,
        ).to(device)

        _pos_ids_set_pt = set(labels_df[labels_df["label"] == 1]["id"])
        _pos_global_idx_pt = np.asarray(
            [i for i, gid in enumerate(target_ids) if gid in _pos_ids_set_pt],
            dtype=np.int64,
        )

        _meta_dec_pt, _ = pretrain_encoder(
            model=_tmp_model,
            meta_t=_meta_t_pt,
            bs_t=_bs_t_pt,
            str_dim=_str_dim_pt,
            device=device,
            pretrain_epochs=pretrain_epochs,
            pretrain_lr=pretrain_lr,
            pretrain_mask_rate=pretrain_mask_rate,
            batch_size=batch_size,
            pos_global_idx=_pos_global_idx_pt,
            w_pretrain_pu=w_pretrain_pu,
            progress_prefix="[PretrainImportance] ",
        )

        top_meta_pairs, top_meta_squares = compute_pretrain_meta_importance(
            model=_tmp_model,
            meta_decoder=_meta_dec_pt,
            meta_t=_meta_t_pt,
            bs_t=_bs_t_pt,
            str_dim=_str_dim_pt,
            device=device,
            meta_col_names=oof_meta.columns.tolist(),
            top_k=poly_top_k,
            batch_size=batch_size,
        )
        del _tmp_model, _meta_dec_pt, _meta_t_pt, _bs_t_pt

        n_new = len(top_meta_pairs) + len(top_meta_squares)
        print(f"[INFO] v25: top features (pretrain MSE): {top_meta_squares}")
        print(
            f"[INFO] v25: {len(top_meta_pairs)} cross-products + "
            f"{len(top_meta_squares)} squares = {n_new} new meta features"
        )

    skf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X_ids = labels_df["id"].values
    y     = labels_df["label"].values

    all_metrics:           list[dict[str, float]] = []
    full_scores_unlabeled: list[pd.DataFrame]     = []
    label_ids_set = set(labels_df["id"].astype(str))
    fold_infos: list[dict] = []

    # v26: global OOF threshold — collect raw (pre-remap) test scores across all folds
    oof_raw_scores: dict[str, float] = {}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_ids, y), start=1):
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df = labels_df.iloc[train_idx].copy().reset_index(drop=True)
        test_df  = labels_df.iloc[test_idx].copy().reset_index(drop=True)
        if label_flip_rate > 0.0:
            _y = train_df["label"].to_numpy(dtype=int)
            train_df["label"] = _apply_label_noise(_y, label_flip_rate, rng)
        train_df.to_csv(fold_dir / "train_labels.tsv", sep="\t", index=False)
        test_df.to_csv( fold_dir / "test_labels.tsv",  sep="\t", index=False)
        test_ids = set(test_df["id"].tolist())

        print(
            f"[INFO] Fold {fold_idx}/{n_splits}: "
            f"n_train={len(train_df)}  n_test={len(test_df)}  "
            f"n_pos_train={int(train_df['label'].sum())}  "
            f"n_neg_train={int((train_df['label']==0).sum())}"
        )

        if string_mode == "anchor":
            string_feature_df = build_fold_string_feature_matrix(
                G=G, target_ids=target_ids,
                anchor_ids=train_df["id"].tolist(),
                max_anchors=max_string_anchors,
            )
        else:
            if string_graph_features is None:
                raise ValueError("string_graph_features is not built")
            string_feature_df = string_graph_features

        meta_all, bs_all, str_all = build_view_frames(
            meta_df=meta_df, brainspan_df=brainspan_df, string_df=string_feature_df,
        )
        if ablate_string:
            str_all = pd.DataFrame(
                np.zeros(str_all.shape, dtype=np.float32),
                index=str_all.index, columns=str_all.columns,
            )
        if ablate_brainspan:
            bs_all = pd.DataFrame(
                np.zeros(bs_all.shape, dtype=np.float32),
                index=bs_all.index, columns=bs_all.columns,
            )

        # Feature noise: applied to training rows of each view
        if noise_type != "none" and noise_level > 0.0:
            _train_ids = train_df["id"].tolist()
            for _view in [meta_all, bs_all, str_all]:
                _arr = _apply_feature_noise(
                    _view.loc[_train_ids].to_numpy(dtype=np.float32),
                    noise_type, noise_level, rng,
                )
                _view.loc[_train_ids] = _arr

        # v25: expand meta_all with XGB-guided polynomial features
        if top_meta_pairs or top_meta_squares:
            meta_all = poly_expand_meta(meta_all, top_meta_pairs, top_meta_squares)

        score_all, feat_all, fit_info = fit_deep_motifs_and_export(
            X_meta_all_raw=meta_all,
            X_bs_all_raw=bs_all,
            X_str_all_raw=str_all,
            ids_all=target_ids,
            train_df=train_df,
            test_ids=test_ids,
            G=G,
            random_state=random_state + fold_idx * 101,
            device=device,
            token_dim=token_dim,
            bs_n_regions=bs_n_regions,
            bs_n_timepoints=bs_n_timepoints,
            str_token_count=str_token_count,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            early_stop_metric=early_stop_metric,
            early_stop_k=early_stop_k,
            augment_factor=augment_factor,
            augment_scale=augment_scale,
            mask_rate_meta=mask_rate_meta,
            mask_rate_bs=mask_rate_bs,
            mask_rate_str=mask_rate_str,
            noise_std=noise_std,
            w_bce=w_bce,
            w_pu=w_pu,
            w_rank=w_rank,
            w_oc=w_oc,
            w_graph=w_graph,
            w_cons=w_cons,
            graph_top_k=graph_top_k,
            xgb_oof_scores=xgb_oof_scores,
            weighted_G=weighted_G,
            gcn_n_layers=gcn_n_layers,
            warmup_epochs=warmup_epochs,
            center_update_interval=center_update_interval,
            use_torch_compile=use_torch_compile,
            pretrain_epochs=pretrain_epochs,
            pretrain_lr=pretrain_lr,
            pretrain_mask_rate=pretrain_mask_rate,
            w_pretrain_pu=w_pretrain_pu,
            ckpt_avg_k=ckpt_avg_k,
            progress_prefix=f"[Fold {fold_idx}][PU] ",
        )

        # v4: 鍥哄畾 alpha=0.5锛屼笉鍐嶅湪璁粌闆嗕笂鎼滅储銆?        # v8: 鏂板 fusion_mode="rrf" 閫夐」锛岀敤 Reciprocal Rank Fusion 浠ｆ浛绾挎€у姞鏉冦€?        
        best_alpha = 0.5
        best_fusion_auc = float("nan")
        if xgb_oof_scores is not None:
            xgb_all = xgb_oof_scores.reindex(score_all.index).fillna(0.5).to_numpy(dtype=float)
            pu_all  = score_all.to_numpy(dtype=float)

            if fusion_mode == "rrf":
                print(
                    f"[Fold {fold_idx}] 铻嶅悎: RRF (k={rrf_k})"
                )
                fused_all = rrf_fuse_scores(xgb_all, pu_all, k=rrf_k)
                best_alpha = -1.0   # sentinel: indicates RRF mode in CSV
            else:
                print(
                    f"[Fold {fold_idx}] 铻嶅悎: fixed alpha=0.50 (XGB鏉冮噸=50%, PU鏉冮噸=50%)"
                )
                fused_all = fuse_scores(xgb_all, pu_all, alpha=best_alpha)

            score_all = pd.Series(fused_all, index=score_all.index, dtype=float)

        # v17: Seeded PPR 鈥?propagate FROM known training positives through STRING
        # v18: save pre-PPR scores for threshold calibration
        # v20: confidence-weighted seeds (XGBoost OOF) + asymmetric PPR fusion weight
        score_all_pre_ppr = score_all.copy()
        if ppr_alpha < 1.0:
            train_pos_ids = train_df[train_df["label"] == 1]["id"].tolist()

            # v20: weight each seed by its XGBoost OOF confidence score
            # High-confidence positives contribute more to PPR 鈫?less noise propagation
            if use_xgb_feature and xgb_oof_scores is not None:
                raw_w   = {gid: float(xgb_oof_scores.get(gid, 0.5)) for gid in train_pos_ids}
                total_w = max(sum(raw_w.values()), 1e-9)
                seed_w  = {k: v / total_w for k, v in raw_w.items()}
            else:
                seed_w  = None

            ppr_scores = compute_ppr_from_seeds(
                seed_ids=train_pos_ids,
                all_ids=list(score_all.index),
                G=weighted_G,
                alpha=ppr_alpha,
                n_iter=ppr_n_iter,
                min_edge_weight=ppr_min_edge_weight,
                seed_weights=seed_w,          # v20: confidence-weighted
            )
            # v20: asymmetric RRF (model ~59%, PPR ~41%) 鈥?preserves right-skewed distribution
            rrf_arr = score_all.to_numpy(dtype=float)
            ppr_arr = ppr_scores.reindex(score_all.index).fillna(0.0).to_numpy(dtype=float)
            final_arr = asymmetric_rrf_fuse(rrf_arr, ppr_arr, k=rrf_k, ppr_w=ppr_fusion_weight)
            score_all = pd.Series(final_arr, index=score_all.index, dtype=float)
            print(
                f"[Fold {fold_idx}] Seeded PPR done  "
                f"n_seeds={len(train_pos_ids)}  alpha={ppr_alpha}  "
                f"min_w={ppr_min_edge_weight}"
            )

        # v19: 娣峰悎鏍″噯 鈥?闃虫€х敤 pre-PPR锛堥伩鍏嶇瀛愯啫鑳€锛夛紝闃存€х敤 post-PPR锛堜笌娴嬭瘯闆嗗悓鍒嗗竷锛?        # 璁粌闃虫€ф槸 PPR 绉嶅瓙锛宲ost-PPR 鍒嗘暟鈮?.0锛堜汉宸ヨ啫鑳€锛夆啋 杩樺師涓?pre-PPR 鍒嗘暟
        # use train-set scores to calibrate threshold consistently with test distribution
        train_ids_fold  = fit_info["train_ids"]
        train_labels    = np.asarray(fit_info["train_labels"], dtype=int)
        if ppr_alpha < 1.0:
            post_arr    = score_all.reindex(train_ids_fold).to_numpy(dtype=float)
            pre_arr     = score_all_pre_ppr.reindex(train_ids_fold).to_numpy(dtype=float)
            pos_mask    = (train_labels == 1)
            fused_train = post_arr.copy()
            fused_train[pos_mask] = pre_arr[pos_mask]   # 绉嶅瓙闃虫€ц繕鍘熶负 pre-PPR 鍒嗘暟
        else:
            fused_train = score_all.reindex(train_ids_fold).to_numpy(dtype=float)
        threshold_final = find_best_threshold_by_f1(train_labels, fused_train)

        # v26: store raw (pre-remap) fused scores for test genes
        for gid, raw_s in zip(
            test_df["id"].tolist(),
            score_all.loc[test_df["id"]].to_numpy(dtype=float).tolist(),
        ):
            oof_raw_scores[gid] = raw_s

        score_all_cal   = pd.Series(
            remap_score_with_threshold(score_all.to_numpy(dtype=float), threshold_final),
            index=score_all.index, dtype=float,
        )

        test_scores  = score_all_cal.loc[test_df["id"]].to_numpy(dtype=float)
        test_metrics = evaluate_predictions(test_df["label"].to_numpy(dtype=int), test_scores)
        test_metrics["fold"]            = fold_idx
        test_metrics["n_test"]          = int(len(test_df))
        test_metrics["fusion_alpha"]    = float(best_alpha)
        test_metrics["pu_contribution"] = float(1.0 - best_alpha)
        all_metrics.append(test_metrics)

        pred_df = test_df.copy()
        pred_df["forecASD"]   = test_scores
        pred_df["pred_label"] = (pred_df["forecASD"] >= 0.5).astype(int)
        pred_df.to_csv(fold_dir / "test_predictions.csv", index=False)
        feat_all.to_csv(fold_dir / "all_gene_component_scores.csv", index=True)

        full_scores_df = pd.DataFrame(
            {"ensembl_string": score_all_cal.index, "forecASD": score_all_cal.values}
        )
        full_scores_df = full_scores_df[
            ~full_scores_df["ensembl_string"].isin(label_ids_set)
        ]
        full_scores_df.to_csv(fold_dir / "full_scores.csv", index=False)
        full_scores_unlabeled.append(full_scores_df.set_index("ensembl_string"))

        fold_info_out = {
            "fold": fold_idx,
            "n_train": len(train_df), "n_test": len(test_df),
            "n_pos_train":  int(train_df["label"].sum()),
            "n_neg_train":  int((train_df["label"] == 0).sum()),
            "n_pos_test":   int(test_df["label"].sum()),
            "n_neg_test":   int((test_df["label"] == 0).sum()),
            "token_dim": token_dim,
            "bs_n_regions": bs_n_regions, "bs_n_timepoints": bs_n_timepoints,
            "str_token_count": str_token_count,
            "transformer_layers": transformer_layers,
            "transformer_heads": transformer_heads,
            "early_stop_metric": early_stop_metric, "early_stop_k": early_stop_k,
            "augment_factor": max(augment_factor, 1),
            "augment_scale": float(max(augment_scale, 0.0)),
            "graph_top_k": graph_top_k,
            "warmup_epochs": warmup_epochs,
            "center_update_interval": center_update_interval,
            "use_xgb_feature": use_xgb_feature,
            "fusion_mode":          fusion_mode,
            "rrf_k":                rrf_k if fusion_mode == "rrf" else None,
            "ppr_alpha":            float(ppr_alpha),
            "ppr_n_iter":           ppr_n_iter,
            "ppr_min_edge_weight":  float(ppr_min_edge_weight),
            "fusion_alpha":         float(best_alpha),
            "fusion_train_pr_auc":  float(best_fusion_auc),
            "best_metric":          float(fit_info["best_metric"]),
            "threshold":            float(fit_info["threshold"]),
            "threshold_fused":      float(threshold_final),
            "n_universe_train":     int(fit_info["n_universe_train"]),
            "n_pos_train_pu":       int(fit_info["n_pos_train"]),
            "n_unlabeled_train_pu": int(fit_info["n_unlabeled_train"]),
        }
        with open(fold_dir / "fold_info.json", "w", encoding="utf-8") as f:
            json.dump(fold_info_out, f, ensure_ascii=False, indent=2)
        fold_infos.append(fold_info_out)

    # ---- v26: Global OOF threshold calibration ----
    print("[INFO] Computing global OOF threshold on full labeled set...")
    label_lookup = labels_df.set_index("id")["label"].to_dict()
    oof_ids_present = [g for g in labels_df["id"].tolist() if g in oof_raw_scores]
    oof_s_arr = np.array([oof_raw_scores[g] for g in oof_ids_present], dtype=float)
    oof_y_arr = np.array([label_lookup[g]   for g in oof_ids_present], dtype=int)
    # beta=0.8: precision-biased F-beta prevents the pooled OOF distribution
    # from driving the threshold too low (which collapses precision vs per-fold).
    global_threshold = find_best_threshold_by_f1(oof_y_arr, oof_s_arr, beta=0.8)
    print(f"[INFO] Global OOF threshold (F-beta=0.8): {global_threshold:.4f}")

    global_metrics_list = []
    for fold_idx_g, (_, test_idx_g) in enumerate(skf.split(X_ids, y), start=1):
        test_df_g  = labels_df.iloc[test_idx_g]
        test_ids_g = test_df_g["id"].tolist()
        test_y_g   = test_df_g["label"].to_numpy(dtype=int)
        test_raw_g = np.array([oof_raw_scores.get(g, 0.5) for g in test_ids_g], dtype=float)
        test_cal_g = remap_score_with_threshold(test_raw_g, global_threshold)
        m_g = evaluate_predictions(test_y_g, test_cal_g)
        m_g["fold"]           = fold_idx_g
        m_g["n_test"]         = len(test_df_g)
        m_g["fusion_alpha"]   = float("nan")
        m_g["pu_contribution"] = float("nan")
        global_metrics_list.append(m_g)

    global_metrics_df = pd.DataFrame(global_metrics_list)
    _global_metric_cols = [c for c in [
        "fold", "n_test",
        "accuracy", "precision", "recall", "f1",
        "macro_f1", "weighted_f1", "pr_auc", "roc_auc",
        "precision@10", "recall@10", "lift@10", "ndcg@10",
        "precision@20", "recall@20", "lift@20", "ndcg@20",
        "precision@50", "recall@50", "lift@50", "ndcg@50",
        "fusion_alpha", "pu_contribution",
    ] if c in global_metrics_df.columns]
    global_metrics_df = global_metrics_df[_global_metric_cols]
    global_metrics_df.to_csv(output_dir / "cv_fold_metrics_global_threshold.csv", index=False)

    global_summary_rows = []
    for col in [c for c in _global_metric_cols if c != "fold"]:
        vals = pd.to_numeric(global_metrics_df[col], errors="coerce")
        global_summary_rows.append({
            "metric": col,
            "mean": float(np.nanmean(vals)),
            "std":  float(np.nanstd(vals, ddof=1)) if vals.notna().sum() > 1 else 0.0,
        })
    pd.DataFrame(global_summary_rows).to_csv(
        output_dir / "cv_metrics_summary_global_threshold.csv", index=False)
    print(f"[INFO] Global threshold={global_threshold:.4f} → cv_metrics_summary_global_threshold.csv")

    metrics_df = pd.DataFrame(all_metrics)
    metric_cols = [
        "fold", "n_test",
        "accuracy", "precision", "recall", "f1",
        "macro_f1", "weighted_f1", "pr_auc", "roc_auc",
        "precision@10", "recall@10", "lift@10", "ndcg@10",
        "precision@20", "recall@20", "lift@20", "ndcg@20",
        "precision@50", "recall@50", "lift@50", "ndcg@50",
        "fusion_alpha", "pu_contribution",
    ]
    metrics_df = metrics_df[metric_cols]
    metrics_df.to_csv(output_dir / "cv_fold_metrics.csv", index=False)

    summary_rows = []
    for col in [c for c in metric_cols if c != "fold"]:
        vals = pd.to_numeric(metrics_df[col], errors="coerce")
        summary_rows.append({
            "metric": col,
            "mean":   float(np.nanmean(vals)),
            "std":    float(np.nanstd(vals, ddof=1)) if vals.notna().sum() > 1 else 0.0,
        })
    pd.DataFrame(summary_rows).to_csv(output_dir / "cv_metrics_summary.csv", index=False)

    if full_scores_unlabeled:
        (pd.concat(full_scores_unlabeled).groupby(level=0).mean(numeric_only=True)
         .reset_index().rename(columns={"index": "ensembl_string"})
         .to_csv(output_dir / "full_scores_summary.csv", index=False))
    if fold_infos:
        pd.DataFrame(fold_infos).to_csv(output_dir / "fold_infos_summary.csv", index=False)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep-MOTIFs — v25 + global OOF threshold calibration.")
    p.add_argument("--project-root",  type=str, required=True)
    p.add_argument("--labels-dir",    type=str, default=None)
    p.add_argument("--output-dir",    type=str, default="deep_motifs_outputs")
    p.add_argument("--tada-filename",    type=str, default="tada_new.csv")
    p.add_argument("--jack-filename",    type=str, default="jack_fu_gene_info(in).csv")
    p.add_argument("--n-splits",      type=int, default=5)
    p.add_argument("--random-state",  type=int, default=42)
    p.add_argument("--string-mode",   type=str, default="anchor",
                   choices=["anchor", "graph"])
    p.add_argument("--max-string-anchors", type=int, default=256)
    p.add_argument("--device",        type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--force-rebuild-brainspan",      action="store_true")
    p.add_argument("--force-rebuild-string",         action="store_true")
    p.add_argument("--force-rebuild-graph-features", action="store_true")
    p.add_argument("--force-rebuild-xgb-oof",        action="store_true")

    # Architecture  (meta_token_count removed 鈥?MetaMLP always gives 1 token)
    p.add_argument("--token-dim",          type=int,   default=64)
    p.add_argument("--bs-n-regions",       type=int,   default=16)
    p.add_argument("--bs-n-timepoints",    type=int,   default=50)
    p.add_argument("--str-token-count",    type=int,   default=6)
    p.add_argument("--transformer-heads",  type=int,   default=4)
    p.add_argument("--transformer-layers", type=int,   default=2)
    p.add_argument("--dropout",            type=float, default=0.3)

    # Training
    p.add_argument("--epochs",             type=int,   default=150)
    p.add_argument("--batch-size",         type=int,   default=128)
    p.add_argument("--learning-rate",      type=float, default=1e-4)
    p.add_argument("--weight-decay",       type=float, default=5e-3)
    p.add_argument("--patience",           type=int,   default=30)
    p.add_argument("--warmup-epochs",      type=int,   default=10)
    p.add_argument("--early-stop-metric",  type=str,   default="pr_auc",
                   choices=["pr_auc", "recall_at_k", "loss"])
    p.add_argument("--early-stop-k",       type=int,   default=50)

    # Augmentation
    p.add_argument("--augment-factor",  type=int,   default=3)
    p.add_argument("--augment-scale",   type=float, default=0.4)
    p.add_argument("--mask-rate-meta",  type=float, default=0.1)
    p.add_argument("--mask-rate-bs",    type=float, default=0.15)
    p.add_argument("--mask-rate-str",   type=float, default=0.15)
    p.add_argument("--noise-std",       type=float, default=0.03)

    # Loss weights  (v2: BCE removed from fine-tune; nnPU is main loss)
    p.add_argument("--w-bce",   type=float, default=0.0)
    p.add_argument("--w-pu",    type=float, default=1.0)
    p.add_argument("--w-rank",  type=float, default=0.5)
    p.add_argument("--w-oc",    type=float, default=0.05)
    p.add_argument("--w-graph", type=float, default=0.05)
    p.add_argument("--w-cons",  type=float, default=0.05)

    # Graph
    p.add_argument("--graph-top-k",   type=int, default=3)
    p.add_argument("--gcn-n-layers",  type=int, default=2,
                   help="Number of GCN aggregation layers (default: 2).")

    # XGBoost OOF
    p.add_argument("--use-xgb-feature",  dest="use_xgb_feature",
                   action="store_true",  default=True)
    p.add_argument("--no-xgb-feature",   dest="use_xgb_feature",
                   action="store_false")
    p.add_argument("--xgb-n-estimators",      type=int,   default=500,
                   help="Number of XGBoost trees (v18 default=500).")
    p.add_argument("--xgb-max-depth",         type=int,   default=4,
                   help="XGBoost max_depth (v18 default=4, was 6 in v8).")
    p.add_argument("--xgb-min-child-weight",  type=int,   default=5,
                   help="XGBoost min_child_weight (v18 default=5, was 1 in v8).")
    p.add_argument("--xgb-reg-alpha",         type=float, default=0.1,
                   help="XGBoost L1 regularisation (v18 default=0.1).")
    p.add_argument("--xgb-gamma",             type=float, default=0.1,
                   help="XGBoost gamma min-split-loss (v18 default=0.1).")

    # Speed
    p.add_argument("--center-update-interval", type=int, default=5)
    p.add_argument("--use-torch-compile",  action="store_true", default=False)

    # v7: Checkpoint Averaging
    p.add_argument("--ckpt-avg-k", type=int, default=5,
                   help="Average the last K improved checkpoints (1 = use single best, no averaging)")

    # v8: Fusion mode
    p.add_argument("--fusion-mode", type=str, default="rrf",
                   choices=["fixed", "rrf"],
                   help="Score fusion strategy: 'fixed' = 0.5脳XGB+0.5脳PU, 'rrf' = Reciprocal Rank Fusion")
    p.add_argument("--rrf-k", type=int, default=60,
                   help="RRF constant k (only used when --fusion-mode=rrf). "
                        "Classic value=60. Smaller 鈫?top ranks dominate more.")

    # v16: Personalized PageRank propagation
    p.add_argument("--ppr-alpha", type=float, default=0.5,
                   help="PPR restart probability. 1.0 = no propagation (same as v8). "
                        "Lower values spread scores more through the STRING network. "
                        "Default=0.5.")
    p.add_argument("--ppr-n-iter", type=int, default=30,
                   help="Max power-iteration steps for PPR (default=30).")
    p.add_argument("--ppr-min-edge-weight", type=float, default=0.5,
                   help="Minimum normalised STRING edge weight to use in propagation. "
                        "0.5 corresponds to STRING confidence score > 700 "
                        "(when base threshold=400). Default=0.5.")
    p.add_argument("--ppr-fusion-weight", type=float, default=0.7,
                   help="v20: PPR multiplier in asymmetric RRF: "
                        "score = 1/(k+rank_model) + ppr_w/(k+rank_ppr). "
                        "1.0 = symmetric RRF (like v18); 0.7 = model ~59%% PPR ~41%%. "
                        "Lower values preserve model precision; higher values boost recall. "
                        "Default=0.7.")

    # v25: XGB-guided polynomial feature expansion
    p.add_argument("--poly-top-k", type=int, default=6,
                   help="v25: top-K meta features by XGB gain importance for polynomial "
                        "expansion (pairwise products + squares). 0 = disabled.")

    # Ablation flags
    p.add_argument("--ablate-string",    action="store_true",
                   help="Ablation: replace all STRING features with zeros (removes graph info).")
    p.add_argument("--ablate-brainspan", action="store_true",
                   help="Ablation: replace all BrainSpan features with zeros (removes temporal info).")
    # Noise robustness
    p.add_argument("--noise-type", type=str, default="none", choices=["none", "gaussian", "dropout"],
                   help="Feature noise type applied to training data only (default: none)")
    p.add_argument("--noise-level", type=float, default=0.0,
                   help="Noise level: std multiplier for gaussian, drop rate for dropout (default: 0.0)")
    p.add_argument("--label-flip-rate", type=float, default=0.0,
                   help="Fraction of negative training labels flipped to positive (default: 0.0)")

    # v6: Masked pre-training
    p.add_argument("--pretrain-epochs",    type=int,   default=50,
                   help="Number of masked feature reconstruction pre-training epochs (0 = skip)")
    p.add_argument("--pretrain-lr",        type=float, default=1e-3,
                   help="Learning rate for pre-training optimizer")
    p.add_argument("--pretrain-mask-rate", type=float, default=0.30,
                   help="Fraction of meta/BrainSpan features to mask during pre-training")
    p.add_argument("--w-pretrain-pu", type=float, default=0.3,
                   help="Weight of nnPU loss during pretrain (v2)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    print(f"[INFO] Using project root: {project_root}")
    data_dir = project_root / "data"
    ensure_exists(data_dir, "data directory")

    labels_dir = (
    Path(args.labels_dir).resolve()
    if args.labels_dir
    else data_dir / "labels"
    )
    output_dir = project_root / args.output_dir

    print("[INFO] Loading composite table...")
    meta_df = load_composite_table(data_dir)

    tada_path = data_dir / args.tada_filename
    jack_path = data_dir / args.jack_filename
    
    print(tada_path)
    print(jack_path)

    ensure_exists(tada_path, args.tada_filename)
    ensure_exists(jack_path, args.jack_filename)
    print("[INFO] Augmenting composite table with tada_new features...")
    meta_df = augment_composite_with_tada(meta_df, tada_path, jack_path)
    print(f"[INFO] Augmented composite table shape: {meta_df.shape}")

    print("[INFO] Loading labels...")
    labels_df = load_labels(labels_dir)
    print("[INFO] Building STRING graph...")
    G = build_string_graph(data_dir, force_rebuild=args.force_rebuild_string)
    print(f"[INFO] STRING graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("[INFO] Building BrainSpan matrix...")
    brainspan_df = build_brainspan_matrix(
        ext_data_dir=data_dir,
        target_proteins=set(meta_df.index.astype(str)),
        force_rebuild=args.force_rebuild_brainspan,
    )
    print(f"[INFO] BrainSpan shape: {brainspan_df.shape}")

    device = resolve_device(args.device)
    fusion_info = (
        f"rrf(k={args.rrf_k})" if args.fusion_mode == "rrf" else "fixed(alpha=0.5)"
    )
    print(
        f"[INFO] Deep-MOTIFs v20 start 鈥?device={device}  "
        f"token_dim={args.token_dim}  "
        f"ckpt_avg_k={args.ckpt_avg_k}  "
        f"early_stop={args.early_stop_metric}  "
        f"pretrain_epochs={args.pretrain_epochs}  "
        f"fusion={fusion_info}  "
        f"ppr_alpha={args.ppr_alpha}  "
        f"use_xgb={args.use_xgb_feature}"
    )

    run_pu(
        labels_df=labels_df,
        meta_df=meta_df,
        brainspan_df=brainspan_df,
        G=G,
        data_dir=data_dir,
        output_dir=output_dir,
        string_mode=args.string_mode,
        max_string_anchors=args.max_string_anchors,
        n_splits=args.n_splits,
        random_state=args.random_state,
        force_rebuild_graph_features=args.force_rebuild_graph_features,
        device=device,
        token_dim=args.token_dim,
        bs_n_regions=args.bs_n_regions,
        bs_n_timepoints=args.bs_n_timepoints,
        str_token_count=args.str_token_count,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        early_stop_metric=args.early_stop_metric,
        early_stop_k=args.early_stop_k,
        augment_factor=args.augment_factor,
        augment_scale=args.augment_scale,
        mask_rate_meta=args.mask_rate_meta,
        mask_rate_bs=args.mask_rate_bs,
        mask_rate_str=args.mask_rate_str,
        noise_std=args.noise_std,
        w_bce=args.w_bce,
        w_pu=args.w_pu,
        w_rank=args.w_rank,
        w_oc=args.w_oc,
        w_graph=args.w_graph,
        w_cons=args.w_cons,
        graph_top_k=args.graph_top_k,
        use_xgb_feature=args.use_xgb_feature,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_min_child_weight=args.xgb_min_child_weight,
        xgb_reg_alpha=args.xgb_reg_alpha,
        xgb_gamma=args.xgb_gamma,
        gcn_n_layers=args.gcn_n_layers,
        warmup_epochs=args.warmup_epochs,
        center_update_interval=args.center_update_interval,
        use_torch_compile=args.use_torch_compile,
        force_rebuild_xgb_oof=args.force_rebuild_xgb_oof,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_lr=args.pretrain_lr,
        pretrain_mask_rate=args.pretrain_mask_rate,
        w_pretrain_pu=args.w_pretrain_pu,
        ckpt_avg_k=args.ckpt_avg_k,
        fusion_mode=args.fusion_mode,
        rrf_k=args.rrf_k,
        ppr_alpha=args.ppr_alpha,
        ppr_n_iter=args.ppr_n_iter,
        ppr_min_edge_weight=args.ppr_min_edge_weight,
        ppr_fusion_weight=args.ppr_fusion_weight,
        poly_top_k=args.poly_top_k,
        ablate_string=args.ablate_string,
        ablate_brainspan=args.ablate_brainspan,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
        label_flip_rate=args.label_flip_rate,
    )
    print(f"[DONE] v26 results saved to: {output_dir}")


if __name__ == "__main__":
    main()

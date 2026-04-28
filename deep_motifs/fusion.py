from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


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
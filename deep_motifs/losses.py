from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


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


def weighted_nnpu_loss(
    pos_logits: torch.Tensor,
    unlabeled_logits: torch.Tensor,
    class_prior: float,
    unlabeled_neg_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Empirical-Bayes responsibility-weighted nnPU.

    For an unlabeled gene u, rho(u) approximates P(hidden positive | evidence).
    The reliable-negative responsibility is therefore w(u)=1-rho(u), and the
    unlabeled negative risk is E_U[w(u) * loss(g(u), 0)].
    """
    if pos_logits.numel() == 0:
        return unlabeled_logits.new_tensor(0.0)
    class_prior = float(np.clip(class_prior, 1e-4, 0.95))
    pos_risk = class_prior * F.binary_cross_entropy_with_logits(
        pos_logits, torch.ones_like(pos_logits)
    )
    if unlabeled_logits.numel() == 0:
        return pos_risk

    unl_neg = F.binary_cross_entropy_with_logits(
        unlabeled_logits, torch.zeros_like(unlabeled_logits), reduction="none"
    )
    if unlabeled_neg_weight is not None:
        w = unlabeled_neg_weight.to(unlabeled_logits.device, dtype=unlabeled_logits.dtype)
        w = w.view_as(unl_neg).clamp(0.0, 1.0)
        unl_neg_risk = torch.mean(w * unl_neg)
    else:
        unl_neg_risk = torch.mean(unl_neg)

    pos_as_neg_risk = F.binary_cross_entropy_with_logits(
        pos_logits, torch.zeros_like(pos_logits)
    )
    neg_risk = unl_neg_risk - class_prior * pos_as_neg_risk
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


def weighted_pairwise_ranking_loss(
    pos_logits: torch.Tensor,
    unlabeled_logits: torch.Tensor,
    unlabeled_neg_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if pos_logits.numel() == 0 or unlabeled_logits.numel() == 0:
        t = pos_logits if pos_logits.numel() > 0 else unlabeled_logits
        return t.new_tensor(0.0)
    diff = pos_logits.unsqueeze(1) - unlabeled_logits.unsqueeze(0)
    loss = F.softplus(-diff)
    if unlabeled_neg_weight is None:
        return loss.mean()
    w = unlabeled_neg_weight.to(unlabeled_logits.device, dtype=unlabeled_logits.dtype)
    w = w.view(1, -1).clamp(0.0, 1.0)
    return (loss * w).sum() / torch.clamp(w.sum() * loss.shape[0], min=1.0)


def rank_percentile_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return scores.astype(np.float32)
    ranks = np.empty(n, dtype=np.float64)
    ranks[np.argsort(scores)] = np.arange(1, n + 1, dtype=np.float64)
    return (ranks / float(n)).astype(np.float32)


def nanmean_std(vals: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(vals, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return float("nan"), float("nan")
    mean = float(valid.mean())
    std = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
    return mean, std

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


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
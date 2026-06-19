from __future__ import annotations

import torch


def make_corrupted_view(
    x: torch.Tensor, mask_rate: float, noise_std: float
) -> torch.Tensor:
    out = x.clone()
    if mask_rate > 0:
        out = out.masked_fill(torch.rand_like(out) < mask_rate, 0.0)
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    return out

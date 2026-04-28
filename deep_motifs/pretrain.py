from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .losses import nnpu_loss
from .models import DeepMOTIFs


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
    import pandas as pd

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
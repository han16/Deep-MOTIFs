from __future__ import annotations

import copy
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import average_precision_score

from .graph import build_neighbor_matrix, _gcn_aggregate_string
from .losses import make_corrupted_view, nnpu_loss, pairwise_ranking_loss
from .metrics import find_best_threshold_by_f1, recall_at_k_score, remap_score_with_threshold
from .models import DeepMOTIFs
from .pretrain import compute_pretrain_meta_importance, pretrain_encoder
from .utils import repeat_array, set_torch_seed, standardize_fit_and_all


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
        # 姣?sp_update_every 涓?epoch锛岀敤褰撳墠妯″瀷瀵?unlabeled 鍩哄洜鎵撳垎锛?        # 鍙繚鐣欏垎鏁颁綆浜?sp_threshold 鐨勫熀鍥犲弬涓?unlabeled loss锛?        # 鎺掗櫎閭ｄ簺妯″瀷璁や负"鍙兘鏄鏍锋湰"鐨勫熀鍥犮€?        # sp_threshold 闅忚缁冭繘琛岀嚎鎬ц"鍑忥紙瓒婃潵瓒婂鏉撅級銆?        if epoch_idx % sp_update_every == 0:
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

            # 姣?5 涓?epoch 鎵嶈"鍑忎竴娆?threshold锛堝叡琛板噺 epochs/5 娆★級
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
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import WeightedRandomSampler

from experiments.xgb import augment_composite_with_tada
from experiments.xgb import build_brainspan_matrix
from experiments.xgb import build_feature_matrix
from experiments.xgb import build_fold_string_feature_matrix
from experiments.xgb import build_string_graph
from experiments.xgb import compute_graph_features
from experiments.xgb import ensure_exists
from experiments.xgb import evaluate_predictions
from experiments.xgb import load_composite_table
from experiments.xgb import load_labels
from experiments.label_noise_utils import (
    add_label_budget_args,
    add_label_noise_args,
    apply_training_label_budget,
    apply_training_label_perturbation,
)


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


def standardize_train_val_all(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    tr = (x_train - mu) / sigma
    va = (x_val - mu) / sigma
    all_x = (x_all - mu) / sigma
    tr = np.clip(np.nan_to_num(tr, nan=0.0, posinf=30.0, neginf=-30.0), -30.0, 30.0)
    va = np.clip(np.nan_to_num(va, nan=0.0, posinf=30.0, neginf=-30.0), -30.0, 30.0)
    all_x = np.clip(np.nan_to_num(all_x, nan=0.0, posinf=30.0, neginf=-30.0), -30.0, 30.0)
    return tr, va, all_x


def build_augmented_view(x: torch.Tensor, mask_rate: float, noise_std: float) -> torch.Tensor:
    out = x
    if mask_rate > 0:
        mask = torch.rand_like(out) < float(mask_rate)
        out = out.masked_fill(mask, 0.0)
    if noise_std > 0:
        out = out + torch.randn_like(out) * float(noise_std)
    return out


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    bsz = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / max(float(temperature), 1e-6)
    eye = torch.eye(2 * bsz, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    targets = torch.arange(2 * bsz, device=z.device)
    targets = (targets + bsz) % (2 * bsz)
    return F.cross_entropy(sim, targets)


class SaintBlock(nn.Module):
    def __init__(self, d_token: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        ff_dim = int(max(d_token * ff_mult, d_token))
        self.col_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.row_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.col_layer(x)
        # Intersample attention: attend over sample dimension for each token.
        xr = x.transpose(0, 1)
        xr = self.row_layer(xr)
        x = xr.transpose(0, 1)
        return x


class SaintClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_token: int,
        n_heads: int,
        n_layers: int,
        ff_mult: int,
        dropout: float,
        proj_dim: int,
    ) -> None:
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError("d_token must be divisible by n_heads")
        self.n_features = int(n_features)
        self.d_token = int(d_token)

        self.feature_w = nn.Parameter(torch.randn(self.n_features, self.d_token) * 0.02)
        self.feature_b = nn.Parameter(torch.zeros(self.n_features, self.d_token))
        self.feature_pos = nn.Parameter(torch.randn(1, self.n_features, self.d_token) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_token))

        self.blocks = nn.ModuleList(
            [SaintBlock(d_token=d_token, n_heads=n_heads, ff_mult=ff_mult, dropout=dropout) for _ in range(n_layers)]
        )
        self.class_head = nn.Sequential(
            nn.LayerNorm(self.d_token),
            nn.Linear(self.d_token, self.d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_token, 1),
        )
        self.proj_head = nn.Sequential(
            nn.LayerNorm(self.d_token),
            nn.Linear(self.d_token, int(proj_dim)),
            nn.GELU(),
            nn.Linear(int(proj_dim), int(proj_dim)),
        )
        self.denoise_head = nn.Sequential(
            nn.LayerNorm(self.d_token),
            nn.Linear(self.d_token, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.feature_w.unsqueeze(0) + self.feature_b.unsqueeze(0)
        tokens = tokens + self.feature_pos
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        for blk in self.blocks:
            seq = blk(seq)
        return seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.encode(x)
        logits = self.class_head(seq[:, 0, :]).squeeze(1)
        return logits

    def project(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.encode(x)
        return self.proj_head(seq[:, 0, :])

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.encode(x)
        rec = self.denoise_head(seq[:, 1:, :]).squeeze(-1)
        return rec


def pretrain_saint(
    model: SaintClassifier,
    x_train: np.ndarray,
    device: torch.device,
    pretrain_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    mask_rate: float,
    noise_std: float,
    nce_temperature: float,
    contrastive_weight: float,
    denoise_weight: float,
    progress_prefix: str,
) -> None:
    if pretrain_epochs <= 0:
        return

    ds = TensorDataset(torch.from_numpy(x_train).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    for epoch_idx in range(pretrain_epochs):
        losses: list[float] = []
        for (xb,) in loader:
            xb = xb.to(device)
            x1 = build_augmented_view(xb, mask_rate=mask_rate, noise_std=noise_std)
            x2 = build_augmented_view(xb, mask_rate=mask_rate, noise_std=noise_std)

            optimizer.zero_grad()
            z1 = model.project(x1)
            z2 = model.project(x2)
            c_loss = info_nce_loss(z1, z2, temperature=nce_temperature)

            rec1 = model.reconstruct(x1)
            rec2 = model.reconstruct(x2)
            d_loss = 0.5 * (F.mse_loss(rec1, xb) + F.mse_loss(rec2, xb))
            loss = float(max(contrastive_weight, 0.0)) * c_loss + float(max(denoise_weight, 0.0)) * d_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite pretraining loss encountered in SAI")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"{progress_prefix}[Pretrain] Epoch {epoch_idx + 1}/{pretrain_epochs} loss={mean_loss:.6f}")


def fit_saint_and_score(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_all: pd.DataFrame,
    random_state: int,
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    dropout: float,
    d_token: int,
    n_heads: int,
    n_layers: int,
    ff_mult: int,
    proj_dim: int,
    feature_mask_rate: float,
    use_balanced_sampling: bool,
    sampling_multiplier: int,
    use_pretrain: bool,
    pretrain_epochs: int,
    pretrain_mask_rate: float,
    pretrain_noise_std: float,
    nce_temperature: float,
    pretrain_contrastive_weight: float,
    pretrain_denoise_weight: float,
    progress_prefix: str = "",
) -> tuple[object, pd.Series]:
    y_train = np.asarray(y_train, dtype=int)
    y_unique = np.unique(y_train)
    if y_unique.size < 2:
        constant = int(y_unique[0]) if y_unique.size == 1 else 0
        scores = np.full(X_all.shape[0], float(constant), dtype=float)
        return None, pd.Series(scores, index=X_all.index)

    set_torch_seed(random_state)

    x_train_np = X_train.to_numpy(dtype=np.float32)
    x_all_np = X_all.to_numpy(dtype=np.float32)

    idx = np.arange(x_train_np.shape[0])
    can_stratify = len(np.unique(y_train)) > 1 and min(np.bincount(y_train)) >= 2
    stratify_arg = y_train if can_stratify else None
    tr_idx, val_idx = train_test_split(
        idx,
        test_size=0.15,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_arg,
    )

    x_tr = x_train_np[tr_idx]
    y_tr = y_train[tr_idx]
    x_val = x_train_np[val_idx]
    y_val = y_train[val_idx]
    x_tr_std, x_val_std, x_all_std = standardize_train_val_all(x_tr, x_val, x_all_np)

    tr_ds = TensorDataset(
        torch.from_numpy(x_tr_std).float(),
        torch.from_numpy(y_tr).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val_std).float(),
        torch.from_numpy(y_val).float(),
    )

    if use_balanced_sampling:
        class_counts = np.bincount(y_tr.astype(int), minlength=2)
        class_counts[class_counts == 0] = 1
        inv = 1.0 / class_counts
        sample_weights = np.clip(inv[y_tr.astype(int)], 1e-6, None)
        num_samples = int(len(sample_weights) * max(sampling_multiplier, 1))
        sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)
        train_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False)
    else:
        train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    model = SaintClassifier(
        n_features=x_tr_std.shape[1],
        d_token=d_token,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_mult=ff_mult,
        dropout=dropout,
        proj_dim=proj_dim,
    ).to(device)

    if use_pretrain and pretrain_epochs > 0:
        pretrain_saint(
            model=model,
            x_train=x_tr_std.astype(np.float32),
            device=device,
            pretrain_epochs=pretrain_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            mask_rate=pretrain_mask_rate,
            noise_std=pretrain_noise_std,
            nce_temperature=nce_temperature,
            contrastive_weight=pretrain_contrastive_weight,
            denoise_weight=pretrain_denoise_weight,
            progress_prefix=progress_prefix,
        )

    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    pos_weight_value = float(n_neg / max(n_pos, 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=learning_rate * 0.1)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    bad_epochs = 0

    for epoch_idx in range(max_epochs):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).clamp(0.0, 1.0)
            if feature_mask_rate > 0:
                mask = torch.rand_like(xb) < float(feature_mask_rate)
                xb = xb.masked_fill(mask, 0.0)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss encountered in SAI")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        val_probs: list[np.ndarray] = []
        val_targets: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).clamp(0.0, 1.0)
                logits = model(xb)
                val_loss = criterion(logits, yb)
                if not torch.isfinite(val_loss):
                    raise RuntimeError("Non-finite validation loss encountered in SAI")
                val_losses.append(float(val_loss.item()))
                val_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
                val_targets.append(yb.detach().cpu().numpy())

        mean_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_probs:
            p = np.concatenate(val_probs, axis=0)
            t = np.concatenate(val_targets, axis=0).astype(int)
            val_f1 = float(f1_score(t, (p >= 0.5).astype(int), zero_division=0))
        else:
            val_f1 = float("nan")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        print(
            f"{progress_prefix}Epoch {epoch_idx + 1}/{max_epochs} "
            f"train_loss={mean_train_loss:.6f} val_loss={mean_val_loss:.6f} "
            f"val_f1={val_f1:.6f} best_val={best_val_loss:.6f} bad_epochs={bad_epochs}/{patience}"
        )
        scheduler.step()
        if bad_epochs >= patience:
            print(f"{progress_prefix}Early stopping at epoch {epoch_idx + 1}/{max_epochs}")
            break

    model.load_state_dict(best_state)
    model.eval()
    scores_out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, x_all_std.shape[0], batch_size):
            xb = torch.from_numpy(x_all_std[i : i + batch_size]).float().to(device)
            probs = torch.sigmoid(model(xb))
            scores_out.append(probs.detach().cpu().numpy())
    scores = np.concatenate(scores_out, axis=0)
    return model, pd.Series(scores, index=X_all.index)


def run_sai(
    labels_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    brainspan_df: pd.DataFrame,
    G,
    output_dir: Path,
    string_mode: str,
    max_string_anchors: int,
    n_splits: int,
    random_state: int,
    force_rebuild_graph_features: bool,
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    dropout: float,
    d_token: int,
    n_heads: int,
    n_layers: int,
    ff_mult: int,
    proj_dim: int,
    feature_mask_rate: float,
    use_balanced_sampling: bool,
    sampling_multiplier: int,
    use_pretrain: bool,
    pretrain_epochs: int,
    pretrain_mask_rate: float,
    pretrain_noise_std: float,
    nce_temperature: float,
    pretrain_contrastive_weight: float,
    pretrain_denoise_weight: float,
    noise_type: str = "none",
    noise_level: float = 0.0,
    label_flip_rate: float = 0.0,
    label_noise_mode: str = "none",
    label_noise_rate: float = 0.0,
    label_budget_positive_fraction: float = 1.0,
    label_budget_neg_ratio: float = 0.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_df = labels_df.reset_index(drop=True)

    target_ids = meta_df.index.astype(str).tolist()
    valid_ids = set(meta_df.index) & set(G.nodes)
    labels_df = labels_df[labels_df["id"].isin(valid_ids)].reset_index(drop=True)

    n_pos = int((labels_df["label"] == 1).sum())
    n_neg = int((labels_df["label"] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "After filtering to IDs present in composite_table and STRING graph, only one class remains. "
            f"Kept n_pos={n_pos}, n_neg={n_neg}."
        )

    labels_df.to_csv(output_dir / "all_labels_used.csv", index=False)

    string_graph_features: pd.DataFrame | None = None
    if string_mode == "graph":
        cache_path = output_dir.parent / "cache" / "string_gene_graph_features.pkl"
        string_graph_features = compute_graph_features(
            G=G,
            target_ids=target_ids,
            cache_path=cache_path,
            force_rebuild=force_rebuild_graph_features,
        )
        string_graph_features = string_graph_features.reindex(meta_df.index)
        string_graph_features = string_graph_features.replace([np.inf, -np.inf], np.nan)
        string_graph_features = string_graph_features.fillna(string_graph_features.median(numeric_only=True)).fillna(0.0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X_ids = labels_df["id"].values
    y = labels_df["label"].values
    rng = np.random.default_rng(random_state)

    all_metrics: list[dict[str, float]] = []
    full_scores_unlabeled: list[pd.DataFrame] = []
    label_ids = set(labels_df["id"].astype(str))

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_ids, y), start=1):
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df = labels_df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = labels_df.iloc[test_idx].copy().reset_index(drop=True)
        train_df, label_noise_info = apply_training_label_perturbation(
            train_df, label_noise_mode, label_noise_rate, random_state + fold_idx * 1009
        )
        train_df, label_budget_info = apply_training_label_budget(
            train_df, label_budget_positive_fraction, label_budget_neg_ratio, random_state + fold_idx * 2003
        )
        train_df.to_csv(fold_dir / "train_labels.tsv", sep="\t", index=False)
        test_df.to_csv(fold_dir / "test_labels.tsv", sep="\t", index=False)

        print(
            f"[INFO] Fold {fold_idx}/{n_splits}: "
            f"n_train={len(train_df)} n_test={len(test_df)} "
            f"n_pos_train={int(train_df['label'].sum())} n_neg_train={int((train_df['label'] == 0).sum())}"
        )

        train_ids = train_df["id"].tolist()
        y_train = train_df["label"].to_numpy(dtype=int)
        y_train = _apply_label_noise(y_train, label_flip_rate, rng)

        if string_mode == "anchor":
            string_feature_df = build_fold_string_feature_matrix(
                G=G,
                target_ids=target_ids,
                anchor_ids=train_ids,
                max_anchors=max_string_anchors,
            )
        else:
            if string_graph_features is None:
                raise ValueError("string_graph_features is not built")
            string_feature_df = string_graph_features

        X_all = build_feature_matrix(
            meta_df=meta_df,
            brainspan_df=brainspan_df,
            string_df=string_feature_df,
        )

        X_train_sai = X_all.loc[train_ids]
        if noise_type != "none" and noise_level > 0.0:
            _arr = _apply_feature_noise(X_train_sai.to_numpy(dtype=np.float32), noise_type, noise_level, rng)
            X_train_sai = pd.DataFrame(_arr, index=X_train_sai.index, columns=X_train_sai.columns)
        _, final_scores = fit_saint_and_score(
            X_train=X_train_sai,
            y_train=y_train,
            X_all=X_all,
            random_state=43775 + fold_idx,
            device=device,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            dropout=dropout,
            d_token=d_token,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_mult=ff_mult,
            proj_dim=proj_dim,
            feature_mask_rate=feature_mask_rate,
            use_balanced_sampling=use_balanced_sampling,
            sampling_multiplier=sampling_multiplier,
            use_pretrain=use_pretrain,
            pretrain_epochs=pretrain_epochs,
            pretrain_mask_rate=pretrain_mask_rate,
            pretrain_noise_std=pretrain_noise_std,
            nce_temperature=nce_temperature,
            pretrain_contrastive_weight=pretrain_contrastive_weight,
            pretrain_denoise_weight=pretrain_denoise_weight,
            progress_prefix=f"[Fold {fold_idx}][SAI] ",
        )

        test_scores = final_scores.loc[test_df["id"]].to_numpy(dtype=float)
        test_metrics = evaluate_predictions(test_df["label"].to_numpy(dtype=int), test_scores)
        test_metrics["fold"] = fold_idx
        test_metrics["n_test"] = int(len(test_df))
        all_metrics.append(test_metrics)

        fold_pred_df = test_df.copy()
        fold_pred_df["forecASD"] = test_scores
        fold_pred_df["pred_label"] = (fold_pred_df["forecASD"] >= 0.5).astype(int)
        fold_pred_df.to_csv(fold_dir / "test_predictions.csv", index=False)

        full_scores_df = pd.DataFrame(
            {
                "gene_id": final_scores.index,
            "ensembl_string": final_scores.index,  # legacy alias for older analysis scripts
                "forecASD": final_scores.values,
            }
        )
        full_scores_df = full_scores_df[~full_scores_df["gene_id"].isin(label_ids)]
        full_scores_df.to_csv(fold_dir / "full_scores.csv", index=False)
        full_scores_unlabeled.append(full_scores_df.set_index("gene_id"))

        with open(fold_dir / "fold_info.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fold": fold_idx,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "n_pos_train": int(train_df["label"].sum()),
                    "n_neg_train": int((train_df["label"] == 0).sum()),
                    "n_pos_test": int(test_df["label"].sum()),
                    "n_neg_test": int((test_df["label"] == 0).sum()),
                    "string_mode": string_mode,
                    "device": str(device),
                    "max_epochs": int(max_epochs),
                    "batch_size": int(batch_size),
                    "learning_rate": float(learning_rate),
                    "weight_decay": float(weight_decay),
                    "patience": int(patience),
                    "dropout": float(dropout),
                    "d_token": int(d_token),
                    "n_heads": int(n_heads),
                    "n_layers": int(n_layers),
                    "ff_mult": int(ff_mult),
                    "proj_dim": int(proj_dim),
                    "feature_mask_rate": float(feature_mask_rate),
                    "use_balanced_sampling": bool(use_balanced_sampling),
                    "sampling_multiplier": int(sampling_multiplier),
                    "use_pretrain": bool(use_pretrain),
                    "pretrain_epochs": int(pretrain_epochs),
                    "pretrain_mask_rate": float(pretrain_mask_rate),
                    "pretrain_noise_std": float(pretrain_noise_std),
                    "nce_temperature": float(nce_temperature),
                    "pretrain_contrastive_weight": float(pretrain_contrastive_weight),
                    "pretrain_denoise_weight": float(pretrain_denoise_weight),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    metrics_df = pd.DataFrame(all_metrics)
    metric_cols = [
        "fold",
        "n_test",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "weighted_f1",
        "pr_auc",
        "roc_auc",
        "precision@10",
        "recall@10",
        "lift@10",
        "ndcg@10",
        "precision@20",
        "recall@20",
        "lift@20",
        "ndcg@20",
        "precision@50",
        "recall@50",
        "lift@50",
        "ndcg@50",
    ]
    metrics_df = metrics_df[metric_cols]
    metrics_df.to_csv(output_dir / "cv_fold_metrics.csv", index=False)

    summary_rows = []
    summary_cols = [c for c in metric_cols if c != "fold"]
    for col in summary_cols:
        vals = pd.to_numeric(metrics_df[col], errors="coerce")
        summary_rows.append(
            {
                "metric": col,
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals, ddof=1)) if vals.notna().sum() > 1 else 0.0,
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "cv_metrics_summary.csv", index=False)

    if full_scores_unlabeled:
        summary_scores = pd.concat(full_scores_unlabeled).groupby(level=0).mean(numeric_only=True)
        summary_scores = summary_scores.reset_index().rename(columns={"index": "gene_id"})
        summary_scores["ensembl_string"] = summary_scores["gene_id"]
        summary_scores.to_csv(output_dir / "full_scores_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAI: SAINT-style (row+column attention + contrastive/denoising pretrain).")
    p.add_argument("--project-root", type=str, required=True, help="Directory containing ext_data/")
    p.add_argument("--labels-dir", type=str, default=None, help="Label directory; defaults to project_root/forecasd_outputs")
    p.add_argument("--output-dir", type=str, default="sai_outputs", help="Output directory name under project root")
    p.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for CV splitting")
    p.add_argument("--string-mode", type=str, default="anchor", choices=["anchor", "graph"], help="STRING feature mode")
    p.add_argument("--max-string-anchors", type=int, default=256, help="Max anchors for string-mode=anchor")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    p.add_argument("--max-epochs", type=int, default=120, help="Supervised max epochs")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size")
    p.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout")
    p.add_argument("--d-token", type=int, default=32, help="Token dimension")
    p.add_argument("--n-heads", type=int, default=4, help="Attention heads")
    p.add_argument("--n-layers", type=int, default=2, help="Number of SAINT blocks")
    p.add_argument("--ff-mult", type=int, default=4, help="Feed-forward expansion")
    p.add_argument("--proj-dim", type=int, default=64, help="Projection head dim for contrastive pretraining")
    p.add_argument("--feature-mask-rate", type=float, default=0.03, help="Feature mask rate in supervised stage")
    p.add_argument("--no-balanced-sampling", action="store_true", help="Disable balanced sampling")
    p.add_argument("--sampling-multiplier", type=int, default=3, help="Sampling multiplier")
    p.add_argument("--no-pretrain", action="store_true", help="Disable SAINT self-supervised pretraining")
    p.add_argument("--pretrain-epochs", type=int, default=30, help="Self-supervised pretraining epochs")
    p.add_argument("--pretrain-mask-rate", type=float, default=0.2, help="Mask rate for pretraining views")
    p.add_argument("--pretrain-noise-std", type=float, default=0.05, help="Noise std for pretraining views")
    p.add_argument("--nce-temperature", type=float, default=0.7, help="Contrastive temperature")
    p.add_argument("--pretrain-contrastive-weight", type=float, default=1.0, help="Contrastive loss weight")
    p.add_argument("--pretrain-denoise-weight", type=float, default=1.0, help="Denoising loss weight")
    p.add_argument("--force-rebuild-brainspan", action="store_true", help="Rebuild cached BrainSpan matrix")
    p.add_argument("--force-rebuild-string", action="store_true", help="Rebuild cached STRING graph")
    p.add_argument("--force-rebuild-graph-features", action="store_true", help="Rebuild cached STRING graph features")
    # Noise robustness
    p.add_argument("--noise-type", type=str, default="none", choices=["none", "gaussian", "dropout"],
                   help="Feature noise type applied to training data only (default: none)")
    p.add_argument("--noise-level", type=float, default=0.0,
                   help="Noise level: std multiplier for gaussian, drop rate for dropout (default: 0.0)")
    p.add_argument("--label-flip-rate", type=float, default=0.0,
                   help="Fraction of negative training labels flipped to positive (default: 0.0)")
    add_label_noise_args(p)
    add_label_budget_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root)
    ext_data_dir = project_root / "ext_data"
    ensure_exists(ext_data_dir, "ext_data directory")

    labels_dir = Path(args.labels_dir).resolve() if args.labels_dir else project_root / "forecasd_outputs"
    output_dir = project_root / args.output_dir

    print("[INFO] Loading composite table...")
    meta_df = load_composite_table(ext_data_dir)
    tada_path = ext_data_dir / "tada_new.csv"
    jack_path = ext_data_dir / "jack_fu_gene_info(in).csv"
    ensure_exists(tada_path, "tada_new.csv")
    ensure_exists(jack_path, "jack_fu_gene_info(in).csv")
    print("[INFO] Augmenting composite table with tada_new features...")
    meta_df = augment_composite_with_tada(meta_df, tada_path, jack_path)
    print(f"[INFO] Augmented composite table shape: {meta_df.shape}")
    print("[INFO] Loading labels...")
    labels_df = load_labels(labels_dir)
    print("[INFO] Building STRING graph...")
    G = build_string_graph(ext_data_dir, force_rebuild=args.force_rebuild_string)
    print(f"[INFO] STRING graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    print("[INFO] Building BrainSpan matrix...")
    brainspan_df = build_brainspan_matrix(
        ext_data_dir=ext_data_dir,
        target_proteins=set(meta_df.index.astype(str)),
        force_rebuild=args.force_rebuild_brainspan,
    )
    print(f"[INFO] BrainSpan matrix shape: {brainspan_df.shape}")

    device = resolve_device(args.device)
    print(f"[INFO] Running SAI (string_mode={args.string_mode}, device={device})...")
    run_sai(
        labels_df=labels_df,
        meta_df=meta_df,
        brainspan_df=brainspan_df,
        G=G,
        output_dir=output_dir,
        string_mode=args.string_mode,
        max_string_anchors=args.max_string_anchors,
        n_splits=args.n_splits,
        random_state=args.random_state,
        force_rebuild_graph_features=args.force_rebuild_graph_features,
        label_noise_mode=args.label_noise_mode,
        label_noise_rate=args.label_noise_rate,
        label_budget_positive_fraction=args.label_budget_positive_fraction,
        label_budget_neg_ratio=args.label_budget_neg_ratio,
        device=device,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        dropout=args.dropout,
        d_token=args.d_token,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_mult=args.ff_mult,
        proj_dim=args.proj_dim,
        feature_mask_rate=args.feature_mask_rate,
        use_balanced_sampling=not args.no_balanced_sampling,
        sampling_multiplier=args.sampling_multiplier,
        use_pretrain=not args.no_pretrain,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_mask_rate=args.pretrain_mask_rate,
        pretrain_noise_std=args.pretrain_noise_std,
        nce_temperature=args.nce_temperature,
        pretrain_contrastive_weight=args.pretrain_contrastive_weight,
        pretrain_denoise_weight=args.pretrain_denoise_weight,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
        label_flip_rate=args.label_flip_rate,
    )
    print(f"[DONE] Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

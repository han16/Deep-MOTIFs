﻿from __future__ import annotations

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
from xgboost import XGBClassifier

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


def fit_xgb_teacher(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_all: pd.DataFrame,
    n_estimators: int,
    random_state: int,
) -> tuple[object, pd.Series]:
    y_train = np.asarray(y_train, dtype=int)
    y_unique = np.unique(y_train)
    if y_unique.size < 2:
        constant = int(y_unique[0]) if y_unique.size == 1 else 0
        scores = np.full(X_all.shape[0], float(constant), dtype=float)
        return None, pd.Series(scores, index=X_all.index)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = float(n_neg / max(n_pos, 1))
    clf = XGBClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        reg_lambda=1.0,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_all)
    if proba.shape[1] == 1:
        scores = np.ones(X_all.shape[0], dtype=float) if int(clf.classes_[0]) == 1 else np.zeros(X_all.shape[0], dtype=float)
    else:
        pos_idx = np.flatnonzero(clf.classes_ == 1)
        scores = proba[:, int(pos_idx[0])] if len(pos_idx) else np.zeros(X_all.shape[0], dtype=float)
    return clf, pd.Series(scores, index=X_all.index)


def build_leaf_encoder(leaf_all: np.ndarray) -> tuple[callable, int]:
    if leaf_all.ndim == 1:
        leaf_all = leaf_all.reshape(-1, 1)
    unique_per_tree: list[np.ndarray] = []
    offsets: list[int] = []
    total_bins = 0
    for t in range(leaf_all.shape[1]):
        uniq = np.unique(leaf_all[:, t].astype(np.int64))
        unique_per_tree.append(uniq)
        offsets.append(total_bins)
        total_bins += int(len(uniq))

    def encode(raw_leaf: np.ndarray) -> np.ndarray:
        arr = raw_leaf
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        out = np.zeros(arr.shape, dtype=np.int64)
        for t, uniq in enumerate(unique_per_tree):
            out[:, t] = np.searchsorted(uniq, arr[:, t].astype(np.int64)) + offsets[t]
        return out

    return encode, int(total_bins)


class CatNNBranch(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.logit = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        l = self.logit(h).squeeze(1)
        return h, l


class GBDT2NNBranch(nn.Module):
    def __init__(self, leaf_vocab_size: int, leaf_emb_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.leaf_embed = nn.Embedding(leaf_vocab_size, leaf_emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(leaf_emb_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.logit = nn.Linear(out_dim, 1)

    def forward(self, leaf_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.leaf_embed(leaf_ids).mean(dim=1)
        h = self.proj(h)
        l = self.logit(h).squeeze(1)
        return h, l


class DeepGBMNet(nn.Module):
    def __init__(
        self,
        n_dense_features: int,
        leaf_vocab_size: int,
        dense_hidden_dim: int,
        branch_dim: int,
        leaf_emb_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cat_branch = CatNNBranch(
            in_dim=n_dense_features,
            hidden_dim=dense_hidden_dim,
            out_dim=branch_dim,
            dropout=dropout,
        )
        self.gbdt_branch = GBDT2NNBranch(
            leaf_vocab_size=leaf_vocab_size,
            leaf_emb_dim=leaf_emb_dim,
            out_dim=branch_dim,
            dropout=dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(branch_dim * 2, branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, 1),
        )

    def forward(self, x_dense: torch.Tensor, leaf_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        h_cat, logit_cat = self.cat_branch(x_dense)
        h_gbdt, logit_gbdt = self.gbdt_branch(leaf_ids)
        logit_final = self.fusion(torch.cat([h_cat, h_gbdt], dim=1)).squeeze(1)
        return {"cat": logit_cat, "gbdt": logit_gbdt, "final": logit_final}


def fit_deepgbm_and_score(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_all: pd.DataFrame,
    random_state: int,
    device: torch.device,
    teacher_n_estimators: int,
    stage1_epochs: int,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    dropout: float,
    dense_hidden_dim: int,
    branch_dim: int,
    leaf_emb_dim: int,
    label_loss_weight: float,
    cat_loss_weight: float,
    gbdt_distill_weight: float,
    final_distill_weight: float,
    feature_mask_rate: float,
    use_balanced_sampling: bool,
    sampling_multiplier: int,
    progress_prefix: str = "",
) -> tuple[object, pd.Series]:
    y_train = np.asarray(y_train, dtype=int)
    y_unique = np.unique(y_train)
    if y_unique.size < 2:
        constant = int(y_unique[0]) if y_unique.size == 1 else 0
        scores = np.full(X_all.shape[0], float(constant), dtype=float)
        return None, pd.Series(scores, index=X_all.index)

    set_torch_seed(random_state)

    idx = np.arange(X_train.shape[0])
    can_stratify = len(np.unique(y_train)) > 1 and min(np.bincount(y_train)) >= 2
    stratify_arg = y_train if can_stratify else None
    tr_idx, val_idx = train_test_split(
        idx,
        test_size=0.15,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_arg,
    )
    train_ids_full = X_train.index.to_numpy()
    tr_ids = train_ids_full[tr_idx]
    val_ids = train_ids_full[val_idx]
    y_tr = y_train[tr_idx]
    y_val = y_train[val_idx]

    teacher_model, teacher_scores_all = fit_xgb_teacher(
        X_train=X_train.loc[tr_ids],
        y_train=y_tr,
        X_all=X_all,
        n_estimators=teacher_n_estimators,
        random_state=random_state,
    )
    if teacher_model is None:
        constant = int(np.unique(y_tr)[0]) if np.unique(y_tr).size == 1 else 0
        scores = np.full(X_all.shape[0], float(constant), dtype=float)
        return None, pd.Series(scores, index=X_all.index)

    x_tr_dense = X_train.loc[tr_ids].to_numpy(dtype=np.float32)
    x_val_dense = X_train.loc[val_ids].to_numpy(dtype=np.float32)
    x_all_dense = X_all.to_numpy(dtype=np.float32)
    x_tr_std, x_val_std, x_all_std = standardize_train_val_all(x_tr_dense, x_val_dense, x_all_dense)

    leaf_all_raw = teacher_model.apply(X_all)
    if leaf_all_raw.ndim == 1:
        leaf_all_raw = leaf_all_raw.reshape(-1, 1)
    encode_leaf, leaf_vocab_size = build_leaf_encoder(leaf_all_raw)
    leaf_all_encoded = encode_leaf(leaf_all_raw)
    leaf_df = pd.DataFrame(leaf_all_encoded, index=X_all.index)
    leaf_tr = leaf_df.loc[tr_ids].to_numpy(dtype=np.int64)
    leaf_val = leaf_df.loc[val_ids].to_numpy(dtype=np.int64)

    teacher_tr = teacher_scores_all.loc[tr_ids].to_numpy(dtype=np.float32)
    teacher_val = teacher_scores_all.loc[val_ids].to_numpy(dtype=np.float32)

    tr_ds = TensorDataset(
        torch.from_numpy(x_tr_std).float(),
        torch.from_numpy(leaf_tr).long(),
        torch.from_numpy(y_tr).float(),
        torch.from_numpy(teacher_tr).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val_std).float(),
        torch.from_numpy(leaf_val).long(),
        torch.from_numpy(y_val).float(),
        torch.from_numpy(teacher_val).float(),
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

    model = DeepGBMNet(
        n_dense_features=x_tr_std.shape[1],
        leaf_vocab_size=leaf_vocab_size,
        dense_hidden_dim=dense_hidden_dim,
        branch_dim=branch_dim,
        leaf_emb_dim=leaf_emb_dim,
        dropout=dropout,
    ).to(device)

    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    pos_weight_value = float(n_neg / max(n_pos, 1))
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=learning_rate * 0.1)

    if stage1_epochs > 0:
        model.train()
        for e in range(stage1_epochs):
            losses: list[float] = []
            for xb, lb, _, tb in train_loader:
                xb = xb.to(device)
                lb = lb.to(device)
                tb = tb.to(device).clamp(0.0, 1.0)
                optimizer.zero_grad()
                out = model(xb, lb)
                loss = F.binary_cross_entropy_with_logits(out["gbdt"], tb)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite Stage-1 loss encountered in DeepGBM")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(float(loss.item()))
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            print(f"{progress_prefix}[Stage1] Epoch {e + 1}/{stage1_epochs} gbdt_mimic_loss={mean_loss:.6f}")

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    bad_epochs = 0

    for epoch_idx in range(max_epochs):
        model.train()
        train_losses: list[float] = []
        for xb, lb, yb, tb in train_loader:
            xb = xb.to(device)
            lb = lb.to(device)
            yb = yb.to(device).clamp(0.0, 1.0)
            tb = tb.to(device).clamp(0.0, 1.0)
            if feature_mask_rate > 0:
                mask = torch.rand_like(xb) < float(feature_mask_rate)
                xb = xb.masked_fill(mask, 0.0)

            optimizer.zero_grad()
            out = model(xb, lb)
            label_loss = cls_criterion(out["final"], yb)
            cat_loss = cls_criterion(out["cat"], yb)
            gbdt_distill = F.binary_cross_entropy_with_logits(out["gbdt"], tb)
            final_distill = F.binary_cross_entropy_with_logits(out["final"], tb)
            loss = (
                float(max(label_loss_weight, 0.0)) * label_loss
                + float(max(cat_loss_weight, 0.0)) * cat_loss
                + float(max(gbdt_distill_weight, 0.0)) * gbdt_distill
                + float(max(final_distill_weight, 0.0)) * final_distill
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss encountered in DeepGBM")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        val_probs: list[np.ndarray] = []
        val_targets: list[np.ndarray] = []
        with torch.no_grad():
            for xb, lb, yb, tb in val_loader:
                xb = xb.to(device)
                lb = lb.to(device)
                yb = yb.to(device).clamp(0.0, 1.0)
                tb = tb.to(device).clamp(0.0, 1.0)
                out = model(xb, lb)
                label_loss = cls_criterion(out["final"], yb)
                cat_loss = cls_criterion(out["cat"], yb)
                gbdt_distill = F.binary_cross_entropy_with_logits(out["gbdt"], tb)
                final_distill = F.binary_cross_entropy_with_logits(out["final"], tb)
                val_loss = (
                    float(max(label_loss_weight, 0.0)) * label_loss
                    + float(max(cat_loss_weight, 0.0)) * cat_loss
                    + float(max(gbdt_distill_weight, 0.0)) * gbdt_distill
                    + float(max(final_distill_weight, 0.0)) * final_distill
                )
                if not torch.isfinite(val_loss):
                    raise RuntimeError("Non-finite validation loss encountered in DeepGBM")
                val_losses.append(float(val_loss.item()))
                val_probs.append(torch.sigmoid(out["final"]).detach().cpu().numpy())
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
    score_out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, x_all_std.shape[0], batch_size):
            xb = torch.from_numpy(x_all_std[i : i + batch_size]).float().to(device)
            lb = torch.from_numpy(leaf_all_encoded[i : i + batch_size]).long().to(device)
            out = model(xb, lb)
            probs = torch.sigmoid(out["final"])
            score_out.append(probs.detach().cpu().numpy())
    scores = np.concatenate(score_out, axis=0)
    return model, pd.Series(scores, index=X_all.index)


def run_deepgbm(
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
    teacher_n_estimators: int,
    stage1_epochs: int,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    dropout: float,
    dense_hidden_dim: int,
    branch_dim: int,
    leaf_emb_dim: int,
    label_loss_weight: float,
    cat_loss_weight: float,
    gbdt_distill_weight: float,
    final_distill_weight: float,
    feature_mask_rate: float,
    use_balanced_sampling: bool,
    sampling_multiplier: int,
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

        train_ids = train_df["id"].tolist()
        y_train = train_df["label"].to_numpy(dtype=int)

        print(
            f"[INFO] Fold {fold_idx}/{n_splits}: "
            f"n_train={len(train_df)} n_test={len(test_df)} "
            f"n_pos_train={int(train_df['label'].sum())} n_neg_train={int((train_df['label'] == 0).sum())}"
        )

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
        X_train = X_all.loc[train_ids]

        _, final_scores = fit_deepgbm_and_score(
            X_train=X_train,
            y_train=y_train,
            X_all=X_all,
            random_state=int(random_state + fold_idx * 137),
            device=device,
            teacher_n_estimators=teacher_n_estimators,
            stage1_epochs=stage1_epochs,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            dropout=dropout,
            dense_hidden_dim=dense_hidden_dim,
            branch_dim=branch_dim,
            leaf_emb_dim=leaf_emb_dim,
            label_loss_weight=label_loss_weight,
            cat_loss_weight=cat_loss_weight,
            gbdt_distill_weight=gbdt_distill_weight,
            final_distill_weight=final_distill_weight,
            feature_mask_rate=feature_mask_rate,
            use_balanced_sampling=use_balanced_sampling,
            sampling_multiplier=sampling_multiplier,
            progress_prefix=f"[Fold {fold_idx}][DeepGBM] ",
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

        full_scores_df = pd.DataFrame({
            "gene_id": final_scores.index,
            "ensembl_string": final_scores.index,  # legacy alias for older analysis scripts
            "forecASD": final_scores.values,
        })
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
                    "teacher_n_estimators": int(teacher_n_estimators),
                    "stage1_epochs": int(stage1_epochs),
                    "max_epochs": int(max_epochs),
                    "batch_size": int(batch_size),
                    "learning_rate": float(learning_rate),
                    "weight_decay": float(weight_decay),
                    "patience": int(patience),
                    "dropout": float(dropout),
                    "dense_hidden_dim": int(dense_hidden_dim),
                    "branch_dim": int(branch_dim),
                    "leaf_emb_dim": int(leaf_emb_dim),
                    "label_loss_weight": float(label_loss_weight),
                    "cat_loss_weight": float(cat_loss_weight),
                    "gbdt_distill_weight": float(gbdt_distill_weight),
                    "final_distill_weight": float(final_distill_weight),
                    "feature_mask_rate": float(feature_mask_rate),
                    "use_balanced_sampling": bool(use_balanced_sampling),
                    "sampling_multiplier": int(sampling_multiplier),
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
    p = argparse.ArgumentParser(description="DeepGBM-like baseline: CatNN + GBDT2NN with distillation.")
    p.add_argument("--project-root", type=str, required=True, help="Directory containing ext_data/")
    p.add_argument("--labels-dir", type=str, default=None, help="Label directory; defaults to project_root/forecasd_outputs")
    p.add_argument("--output-dir", type=str, default="deepgbm_outputs", help="Output directory name under project root")
    p.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for CV splitting")
    p.add_argument("--string-mode", type=str, default="anchor", choices=["anchor", "graph"], help="STRING feature mode")
    p.add_argument("--max-string-anchors", type=int, default=256, help="Max anchors for string-mode=anchor")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    p.add_argument("--teacher-n-estimators", type=int, default=300, help="XGBoost trees for teacher model")
    p.add_argument("--stage1-epochs", type=int, default=20, help="Stage-1 epochs for GBDT2NN distillation warmup")
    p.add_argument("--max-epochs", type=int, default=120, help="Stage-2 max epochs")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    p.add_argument("--dropout", type=float, default=0.25, help="Dropout")
    p.add_argument("--dense-hidden-dim", type=int, default=512, help="Dense branch hidden dim")
    p.add_argument("--branch-dim", type=int, default=128, help="Per-branch representation dim")
    p.add_argument("--leaf-emb-dim", type=int, default=16, help="Leaf embedding dim")
    p.add_argument("--label-loss-weight", type=float, default=1.0, help="Weight for final supervised loss")
    p.add_argument("--cat-loss-weight", type=float, default=0.4, help="Weight for dense branch supervised loss")
    p.add_argument("--gbdt-distill-weight", type=float, default=0.8, help="Weight for GBDT branch teacher distillation")
    p.add_argument("--final-distill-weight", type=float, default=0.2, help="Weight for final output teacher distillation")
    p.add_argument("--feature-mask-rate", type=float, default=0.02, help="Dense feature masking rate during train")
    p.add_argument("--no-balanced-sampling", action="store_true", help="Disable balanced sampling")
    p.add_argument("--sampling-multiplier", type=int, default=3, help="Sampling multiplier")
    p.add_argument("--force-rebuild-brainspan", action="store_true", help="Rebuild cached BrainSpan matrix")
    p.add_argument("--force-rebuild-string", action="store_true", help="Rebuild cached STRING graph")
    p.add_argument("--force-rebuild-graph-features", action="store_true", help="Rebuild cached STRING graph features")
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
    print(f"[INFO] Running DeepGBM-like baseline (string_mode={args.string_mode}, device={device})...")
    run_deepgbm(
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
        teacher_n_estimators=args.teacher_n_estimators,
        stage1_epochs=args.stage1_epochs,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        dropout=args.dropout,
        dense_hidden_dim=args.dense_hidden_dim,
        branch_dim=args.branch_dim,
        leaf_emb_dim=args.leaf_emb_dim,
        label_loss_weight=args.label_loss_weight,
        cat_loss_weight=args.cat_loss_weight,
        gbdt_distill_weight=args.gbdt_distill_weight,
        final_distill_weight=args.final_distill_weight,
        feature_mask_rate=args.feature_mask_rate,
        use_balanced_sampling=not args.no_balanced_sampling,
        sampling_multiplier=args.sampling_multiplier,
    )
    print(f"[DONE] Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

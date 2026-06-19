﻿from __future__ import annotations

# FT-Transformer baseline 鈥?Feature Tokenizer + Transformer for tabular features.
# Uses meta features only (cols i>7 from composite_table, ~11-23 features).
# BrainSpan (800-d) and STRING features are NOT used to keep tokenization tractable.

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.label_noise_utils import (
    add_label_budget_args,
    add_label_noise_args,
    apply_training_label_budget,
    apply_training_label_perturbation,
)
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Import shared utilities from xgb.py
sys.path.insert(0, str(Path(__file__).parent))
from experiments.xgb import (
    augment_composite_with_tada,
    build_brainspan_matrix,
    build_string_graph,
    coerce_numeric_and_impute,
    ensure_exists,
    evaluate_predictions,
    load_composite_table,
    load_labels,
)


# ============================================================
# Model definition (Gorishniy et al. 2021 FT-Transformer)
# ============================================================

class FeatureTokenizer(nn.Module):
    """
    Tokenizes each scalar feature into a d_token-dimensional embedding.
    Implements: token_i = x_i * W_i + b_i  for each feature i.
    """

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token))
        self.bias = nn.Parameter(torch.randn(n_features, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features)
        # output: (B, n_features, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer.

    Architecture:
      1. FeatureTokenizer: (B, F) 鈫?(B, F, d_token)
      2. Prepend [CLS] token: (B, F+1, d_token)
      3. TransformerEncoder (Pre-LN): n_layers of attention
      4. CLS output 鈫?classification head 鈫?logit
    """

    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features=n_features, d_token=d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=nhead,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN (as in original FT-Transformer)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.ReLU(),
            nn.Linear(d_token // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.tokenizer(x)  # (B, n_features, d_token)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_token)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, n_features+1, d_token)
        out = self.transformer(tokens)  # (B, n_features+1, d_token)
        cls_out = out[:, 0, :]  # CLS token output: (B, d_token)
        return self.classifier(cls_out)  # (B, 1)


# ============================================================
# Feature construction (meta only)
# ============================================================

def build_meta_features(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numeric meta features (cols i>7, 1-indexed) from composite_table.
    Does NOT include BrainSpan or STRING.
    """
    meta_cols = meta_df.columns.tolist()
    keep_meta_cols = [c for i, c in enumerate(meta_cols, start=1) if i > 7]
    work_meta = meta_df[keep_meta_cols] if keep_meta_cols else meta_df.copy()
    return coerce_numeric_and_impute(work_meta)


# ============================================================
# Utility
# ============================================================

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ============================================================
# Training helpers
# ============================================================

def train_ftt(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    n_features: int,
    device: torch.device,
    d_token: int,
    n_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    random_state: int,
) -> FTTransformer:
    """Train FTTransformer with gradient clipping and early stopping on val PR-AUC."""
    from sklearn.metrics import average_precision_score

    # 10% stratified validation split from training set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
    val_split = list(sss.split(X_train_np, y_train_np))
    tr_idx, va_idx = val_split[0]
    X_tr, y_tr = X_train_np[tr_idx], y_train_np[tr_idx]
    X_va, y_va = X_train_np[va_idx], y_train_np[va_idx]

    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)

    # Determine nhead: largest power-of-2 divisor of d_token, max 8
    nhead = 1
    for h in [8, 4, 2, 1]:
        if d_token % h == 0:
            nhead = h
            break

    model = FTTransformer(
        n_features=n_features,
        d_token=d_token,
        n_layers=n_layers,
        nhead=nhead,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    y_va_np = y_va.astype(int)

    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(random_state),
    )

    best_val_prauc = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t).squeeze(1).cpu().numpy()
        val_scores = 1.0 / (1.0 + np.exp(-val_logits))

        if len(np.unique(y_va_np)) == 2:
            val_prauc = float(average_precision_score(y_va_np, val_scores))
        else:
            val_prauc = 0.0

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  [FTT] Early stopping at epoch {epoch + 1}, best val PR-AUC={best_val_prauc:.4f}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model


def score_ftt(
    model: FTTransformer,
    X_np: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    all_scores = []
    with torch.no_grad():
        for start in range(0, X_t.shape[0], batch_size):
            xb = X_t[start: start + batch_size]
            logits = model(xb).squeeze(1)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)
    return np.concatenate(all_scores)


# ============================================================
# Main pipeline
# ============================================================

def run_ftt(
    labels_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    G,
    output_dir: Path,
    n_splits: int,
    random_state: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    d_token: int,
    n_layers: int,
    dropout: float,
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

    # Build meta-only feature matrix (no BrainSpan, no STRING)
    X_meta = build_meta_features(meta_df)
    print(f"[FTT] Meta feature matrix shape: {X_meta.shape} (meta-only, no BrainSpan/STRING)")

    label_ids = set(labels_df["id"].astype(str))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X_ids = labels_df["id"].values
    y = labels_df["label"].values

    all_metrics: list[dict] = []
    full_scores_unlabeled: list[pd.DataFrame] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_ids, y), start=1):
        print(f"[FTT] Fold {fold_idx}/{n_splits}")
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

        # Standardize: fit on train, apply to all
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_meta.loc[train_ids].to_numpy(dtype=np.float32))
        X_all_scaled = scaler.transform(X_meta.to_numpy(dtype=np.float32))

        n_features = X_all_scaled.shape[1]

        model = train_ftt(
            X_train_np=X_train_scaled,
            y_train_np=y_train,
            n_features=n_features,
            device=device,
            d_token=d_token,
            n_layers=n_layers,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            random_state=random_state + fold_idx,
        )

        all_scores_np = score_ftt(model, X_all_scaled, device)
        final_scores = pd.Series(all_scores_np, index=X_meta.index)

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
                    "n_features": n_features,
                    "d_token": d_token,
                    "n_layers": n_layers,
                    "feature_set": "meta_only",
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


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "FT-Transformer baseline 鈥?Feature Tokenizer + Transformer for tabular features. "
            "Uses meta features only (cols i>7 from composite_table, ~11-23 features). "
            "BrainSpan (800-d) and STRING features are excluded to keep tokenization tractable."
        )
    )
    p.add_argument("--project-root", type=str, required=True, help="Directory containing ext_data/")
    p.add_argument("--labels-dir", type=str, default=None,
                   help="Label directory; defaults to project_root/forecasd_outputs")
    p.add_argument("--output-dir", type=str, default="ftt_outputs",
                   help="Output directory name under project root")
    p.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for CV splitting")
    p.add_argument("--string-mode", type=str, default="anchor", choices=["anchor", "graph"],
                   help="STRING feature mode (unused for FTT; meta features only)")
    p.add_argument("--max-string-anchors", type=int, default=256,
                   help="Max anchors (unused for FTT)")
    p.add_argument("--force-rebuild-brainspan", action="store_true",
                   help="(Unused for FTT 鈥?BrainSpan not loaded)")
    p.add_argument("--force-rebuild-string", action="store_true",
                   help="Rebuild cached STRING graph (used only for label filtering)")
    p.add_argument("--force-rebuild-graph-features", action="store_true",
                   help="(Unused for FTT)")
    # Neural network specific
    p.add_argument("--device", type=str, default="auto",
                   help="Device: auto / cpu / cuda (default=auto)")
    p.add_argument("--epochs", type=int, default=300, help="Max training epochs (default=300)")
    p.add_argument("--batch-size", type=int, default=128, help="Mini-batch size (default=128)")
    p.add_argument("--learning-rate", type=float, default=1e-4,
                   help="AdamW learning rate (default=1e-4)")
    p.add_argument("--d-token", type=int, default=64,
                   help="Token embedding dimension (default=64)")
    p.add_argument("--n-layers", type=int, default=3,
                   help="Number of Transformer encoder layers (default=3)")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default=0.1)")
    p.add_argument("--patience", type=int, default=30,
                   help="Early stopping patience on val PR-AUC (default=30)")
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
    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")

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

    # STRING graph is loaded only to filter labels to valid IDs (same as other baselines)
    print("[INFO] Building STRING graph (for label filtering)...")
    G = build_string_graph(ext_data_dir, force_rebuild=args.force_rebuild_string)
    print(f"[INFO] STRING graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

    print("[INFO] Running FT-Transformer baseline (meta features only)...")
    run_ftt(
        labels_df=labels_df,
        meta_df=meta_df,
        G=G,
        output_dir=output_dir,
        n_splits=args.n_splits,
        random_state=args.random_state,
        label_noise_mode=args.label_noise_mode,
        label_noise_rate=args.label_noise_rate,
        label_budget_positive_fraction=args.label_budget_positive_fraction,
        label_budget_neg_ratio=args.label_budget_neg_ratio,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        patience=args.patience,
        d_token=args.d_token,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )

    print(f"[DONE] Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

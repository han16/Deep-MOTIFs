﻿from __future__ import annotations

# DeepASDPred-style CNN-LSTM baseline for ASD gene prioritisation.

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
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

# Import shared utilities from xgb.py
sys.path.insert(0, str(Path(__file__).parent))
from experiments.xgb import (
    augment_composite_with_tada,
    build_brainspan_matrix,
    build_feature_matrix,
    build_fold_string_feature_matrix,
    build_string_graph,
    coerce_numeric_and_impute,
    compute_graph_features,
    ensure_exists,
    evaluate_predictions,
    load_composite_table,
    load_labels,
)


# ============================================================
# Model definition
# ============================================================

class DeepASDPredCNNLSTM(nn.Module):
    """CNN-LSTM baseline following the DeepASDPred classifier design.

    The cited model uses a 1D CNN branch and an LSTM branch in parallel,
    followed by concatenation and dense classification. The reported CNN
    parameters are: one convolutional layer, 64 filters, kernel size 3,
    stride 1, learning rate 1e-4, and batch size 64.
    """

    def __init__(
        self,
        n_features: int,
        n_filters: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        lstm_hidden: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.cnn_pool = nn.AdaptiveMaxPool1d(1)

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        concat_dim = n_filters + lstm_hidden
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(concat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN branch: (B, F) -> (B, 1, F) -> (B, n_filters)
        x_cnn = x.unsqueeze(1)
        h_cnn = self.relu(self.conv(x_cnn))
        h_cnn = self.cnn_pool(h_cnn).squeeze(-1)

        # LSTM branch: treat the selected feature vector as a 1D sequence.
        x_seq = x.unsqueeze(-1)
        _, (h_n, _) = self.lstm(x_seq)
        h_lstm = h_n[-1]

        h = torch.cat([h_cnn, h_lstm], dim=1)
        return self.classifier(h)


# ============================================================
# Training helpers
# ============================================================

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def train_cnn(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    n_features: int,
    device: torch.device,
    n_filters: int,
    kernel_size: int,
    stride: int,
    lstm_hidden: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    random_state: int,
) -> DeepASDPredCNNLSTM:
    """Train DeepASDPred-style CNN-LSTM with early stopping on validation PR-AUC."""
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

    model = DeepASDPredCNNLSTM(
        n_features=n_features,
        n_filters=n_filters,
        kernel_size=kernel_size,
        stride=stride,
        lstm_hidden=lstm_hidden,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
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
            print(f"  [CNN] Early stopping at epoch {epoch + 1}, best val PR-AUC={best_val_prauc:.4f}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model


def score_cnn(
    model: DeepASDPredCNNLSTM,
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

def run_cnn(
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
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    n_filters: int,
    kernel_size: int,
    stride: int,
    lstm_hidden: int,
    dropout: float,
    feature_select_ratio: float,
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
        string_graph_features = string_graph_features.fillna(
            string_graph_features.median(numeric_only=True)
        ).fillna(0.0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X_ids = labels_df["id"].values
    y = labels_df["label"].values

    all_metrics: list[dict] = []
    full_scores_unlabeled: list[pd.DataFrame] = []
    label_ids = set(labels_df["id"].astype(str))

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_ids, y), start=1):
        print(f"[CNN] Fold {fold_idx}/{n_splits}")
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

        # DeepASDPred-style preprocessing: max-min normalization followed by
        # chi-square feature selection. Fit only on the training split.
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_all.loc[train_ids].to_numpy(dtype=np.float32))
        X_all_scaled = scaler.transform(X_all.to_numpy(dtype=np.float32))
        X_train_scaled = np.clip(X_train_scaled, 0.0, 1.0)
        X_all_scaled = np.clip(X_all_scaled, 0.0, 1.0)

        if feature_select_ratio > 0:
            k = max(1, int(round(X_train_scaled.shape[1] * feature_select_ratio)))
            k = min(k, X_train_scaled.shape[1])
            selector = SelectKBest(score_func=chi2, k=k)
            X_train_scaled = selector.fit_transform(X_train_scaled, y_train)
            X_all_scaled = selector.transform(X_all_scaled)
        else:
            k = X_train_scaled.shape[1]

        n_features = X_all_scaled.shape[1]

        model = train_cnn(
            X_train_np=X_train_scaled,
            y_train_np=y_train,
            n_features=n_features,
            device=device,
            n_filters=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            lstm_hidden=lstm_hidden,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            random_state=random_state + fold_idx,
        )

        all_scores_np = score_cnn(model, X_all_scaled, device)
        final_scores = pd.Series(all_scores_np, index=X_all.index)

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
                    "n_features": n_features,
                    "feature_select_ratio": float(feature_select_ratio),
                    "n_selected_features": int(k),
                    "cnn_filters": int(n_filters),
                    "cnn_kernel_size": int(kernel_size),
                    "cnn_stride": int(stride),
                    "lstm_hidden": int(lstm_hidden),
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
        description="DeepASDPred-style CNN-LSTM baseline for ASD gene prioritisation."
    )
    p.add_argument("--project-root", type=str, required=True, help="Directory containing ext_data/")
    p.add_argument("--labels-dir", type=str, default=None,
                   help="Label directory; defaults to project_root/forecasd_outputs")
    p.add_argument("--output-dir", type=str, default="cnn_outputs",
                   help="Output directory name under project root")
    p.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for CV splitting")
    p.add_argument("--string-mode", type=str, default="anchor", choices=["anchor", "graph"],
                   help="STRING feature mode")
    p.add_argument("--max-string-anchors", type=int, default=256,
                   help="Max anchors for string-mode=anchor")
    p.add_argument("--force-rebuild-brainspan", action="store_true",
                   help="Rebuild cached BrainSpan matrix")
    p.add_argument("--force-rebuild-string", action="store_true",
                   help="Rebuild cached STRING graph")
    p.add_argument("--force-rebuild-graph-features", action="store_true",
                   help="Rebuild cached STRING graph features")
    # Neural network specific
    p.add_argument("--device", type=str, default="auto",
                   help="Device: auto / cpu / cuda (default=auto)")
    p.add_argument("--epochs", type=int, default=200, help="Max training epochs (default=200)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Mini-batch size; DeepASDPred optimal CNN value is 64")
    p.add_argument("--learning-rate", type=float, default=1e-4,
                   help="Adam learning rate; DeepASDPred optimal CNN value is 1e-4")
    p.add_argument("--patience", type=int, default=25,
                   help="Early stopping patience on val PR-AUC (default=25)")
    p.add_argument("--n-filters", type=int, default=64,
                   help="Number of convolution filters; DeepASDPred optimal value is 64")
    p.add_argument("--kernel-size", type=int, default=3,
                   help="1D CNN kernel size; DeepASDPred optimal value is 3")
    p.add_argument("--stride", type=int, default=1,
                   help="1D CNN stride; DeepASDPred optimal value is 1")
    p.add_argument("--lstm-hidden", type=int, default=64,
                   help="Hidden units in the parallel LSTM branch")
    p.add_argument("--dropout", type=float, default=0.3, help="Dropout rate (default=0.3)")
    p.add_argument("--feature-select-ratio", type=float, default=0.103,
                   help="Fraction of features retained by chi-square selection; "
                        "DeepASDPred reported 10.3%% as the best subset")
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

    print(f"[INFO] Running CNN baseline (string_mode={args.string_mode})...")
    run_cnn(
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        patience=args.patience,
        n_filters=args.n_filters,
        kernel_size=args.kernel_size,
        stride=args.stride,
        lstm_hidden=args.lstm_hidden,
        dropout=args.dropout,
        feature_select_ratio=args.feature_select_ratio,
    )

    print(f"[DONE] Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

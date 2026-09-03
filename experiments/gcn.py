from __future__ import annotations

# DeepND-style GCN baseline on STRING PPI graph.
#
# This keeps the original input/output contract of this project, but replaces
# the plain 2-layer GCN with a DeepND-ST-inspired architecture:
# multiple graph-convolution experts plus a gene-level gating network.

import argparse
import gzip
import json
import sys
from pathlib import Path

import networkx as nx
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
    save_pickle,
    load_pickle,
)


# ============================================================
# Model definition
# ============================================================

class GCNConv(nn.Module):
    """Single GCN convolutional layer: A_norm @ (X @ W) + b."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, adj_norm: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        support = self.W(x)  # (N, out_dim)
        out = torch.sparse.mm(adj_norm, support) + self.bias  # (N, out_dim)
        return out


class GCN(nn.Module):
    """2-layer GCN with ReLU activations and dropout."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 64)
        self.classifier = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj_norm: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(adj_norm, x))
        h = self.dropout(h)
        h = self.conv2(adj_norm, h)
        return self.classifier(h)  # (N, 1)


class DeepNDExpert(nn.Module):
    """
    One DeepND-ST graph expert.

    DeepND uses a 2-layer GCN expert for each co-expression network. In the
    original model the first graph-convolution layer has dimension 4 and the
    second layer outputs a one-dimensional positive-class score. We keep that
    structure, but use our project's gene feature matrix and STRING-derived
    expert graphs.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 4, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj_norm: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(adj_norm, x))
        h = self.dropout(h)
        return self.conv2(adj_norm, h).squeeze(1)


class DeepNDStyleGCN(nn.Module):
    """
    DeepND-ST-inspired mixture of graph-convolution experts.

    Each expert processes one STRING-derived graph. The gating network receives
    node features and assigns a softmax weight to each expert for each gene.
    Final risk probability is the weighted sum of expert probabilities.
    """

    def __init__(
        self,
        in_dim: int,
        n_experts: int,
        hidden_dim: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [DeepNDExpert(in_dim, hidden_dim=hidden_dim, dropout=dropout)
             for _ in range(n_experts)]
        )
        self.gate = nn.Linear(in_dim, n_experts)

    def forward(
        self,
        adj_norms: list[torch.Tensor],
        x: torch.Tensor,
        return_parts: bool = False,
    ):
        expert_logits = torch.stack(
            [expert(adj, x) for expert, adj in zip(self.experts, adj_norms)],
            dim=1,
        )
        expert_probs = torch.sigmoid(expert_logits)
        gate_weights = torch.softmax(self.gate(x), dim=1)
        prob = torch.sum(gate_weights * expert_probs, dim=1)
        if return_parts:
            return prob, expert_probs, gate_weights
        return prob


# ============================================================
# Graph utilities
# ============================================================

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def build_normalized_adjacency(
    G: nx.Graph,
    node_list: list[str],
) -> torch.Tensor:
    """
    Build sparse normalized adjacency A_norm = D^(-1/2) (A + I) D^(-1/2).

    Edge weights from STRING 'weight' attribute are used when available;
    edges without a weight attribute default to 1.0.
    Self-loops are added before normalization.
    """
    n = len(node_list)
    node_idx = {node: i for i, node in enumerate(node_list)}

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    # Add edges with weights
    for u, v, data in G.edges(data=True):
        i = node_idx.get(u)
        j = node_idx.get(v)
        if i is None or j is None:
            continue
        w = float(data.get("weight", 1.0))
        rows.append(i)
        cols.append(j)
        vals.append(w)
        if i != j:
            rows.append(j)
            cols.append(i)
            vals.append(w)

    # Add self-loops (weight=1.0)
    for i in range(n):
        rows.append(i)
        cols.append(i)
        vals.append(1.0)

    # Build degree vector from (A + I)
    deg = np.zeros(n, dtype=np.float64)
    for i, v in zip(rows, vals):
        deg[i] += v

    # Normalize: D^(-1/2) (A+I) D^(-1/2)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    norm_vals = [d_inv_sqrt[r] * v * d_inv_sqrt[c] for r, c, v in zip(rows, cols, vals)]

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(norm_vals, dtype=torch.float32)
    adj_norm = torch.sparse_coo_tensor(indices, values, size=(n, n))
    return adj_norm.coalesce()


def parse_expert_thresholds(thresholds: str) -> list[int]:
    out: list[int] = []
    for item in str(thresholds).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(float(item)))
    if not out:
        out = [400]
    return sorted(set(out))


def build_string_threshold_graphs(
    ext_data_dir: Path,
    thresholds: list[int],
    node_list: list[str],
) -> list[nx.Graph]:
    """
    Build one STRING expert graph per score threshold.

    DeepND uses multiple brain co-expression networks as experts. Our inputs do
    not include those 52 networks, so we keep the same public input contract and
    derive multiple experts from STRING by confidence thresholds.
    """
    string_path = ext_data_dir / "9606.protein.links.v10.txt.gz"
    ensure_exists(string_path, "9606.protein.links.v10.txt.gz")
    node_set = set(node_list)
    graphs = []
    for threshold in thresholds:
        Gt = nx.Graph()
        Gt.add_nodes_from(node_list)
        with gzip.open(string_path, "rt") as f:
            _ = next(f)
            for line in f:
                a, b, score_s = line.strip().split()
                score = int(score_s)
                if score <= threshold:
                    continue
                a = a.replace("9606.", "")
                b = b.replace("9606.", "")
                if a in node_set and b in node_set:
                    Gt.add_edge(a, b, weight=float(score) / 1000.0)
        graphs.append(Gt)
    return graphs


def build_gcn_node_features(
    meta_df: pd.DataFrame,
    brainspan_df: pd.DataFrame,
    node_list: list[str],
    use_brainspan: bool = True,
) -> np.ndarray:
    """
    Build node feature matrix for all nodes in node_list.
    Uses meta features (cols i>7) + BrainSpan (if use_brainspan=True).
    Nodes not in meta_df receive zero features.
    """
    meta_cols = meta_df.columns.tolist()
    keep_meta_cols = [c for i, c in enumerate(meta_cols, start=1) if i > 7]
    work_meta = meta_df[keep_meta_cols] if keep_meta_cols else meta_df.copy()
    meta_num = coerce_numeric_and_impute(work_meta)

    if use_brainspan and brainspan_df is not None and brainspan_df.shape[1] > 0:
        bs_median = brainspan_df.median(numeric_only=True)
        bs_all = brainspan_df.reindex(meta_df.index).fillna(bs_median)
        bs_all = bs_all.replace([np.inf, -np.inf], np.nan)
        bs_all = bs_all.fillna(bs_all.median(numeric_only=True)).fillna(0.0)
        feat_df = pd.concat([meta_num, bs_all], axis=1)
    else:
        feat_df = meta_num

    feat_df = coerce_numeric_and_impute(feat_df)
    n_feat = feat_df.shape[1]

    # Build feature matrix aligned to node_list order, zero-fill missing
    feat_matrix = np.zeros((len(node_list), n_feat), dtype=np.float32)
    for i, node in enumerate(node_list):
        if node in feat_df.index:
            feat_matrix[i] = feat_df.loc[node].to_numpy(dtype=np.float32)

    return feat_matrix


# ============================================================
# Training helpers
# ============================================================

def weighted_bce_from_probs(
    probs: torch.Tensor,
    y_true: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
    loss = -(pos_weight * y_true * torch.log(probs) + (1.0 - y_true) * torch.log(1.0 - probs))
    return loss.mean()


def train_gcn(
    adj_norms: list[torch.Tensor],
    X_all_t: torch.Tensor,
    node_idx_map: dict[str, int],
    train_ids: list[str],
    y_train: np.ndarray,
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    random_state: int,
) -> DeepNDStyleGCN:
    """
    Transductive DeepND-style GCN training.
    Forward pass runs on ALL nodes; loss is computed only on labeled training nodes.
    Early stopping monitors val PR-AUC on 10% of labeled train nodes.
    """
    from sklearn.metrics import average_precision_score

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())

    # 10% stratified validation split from training labeled nodes
    if len(np.unique(y_train)) == 2 and len(y_train) >= 10:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
        splits = list(sss.split(train_ids, y_train))
        tr_sub_idx, va_sub_idx = splits[0]
    else:
        tr_sub_idx = np.arange(len(train_ids))
        va_sub_idx = np.arange(len(train_ids))

    tr_node_indices = [node_idx_map[nid] for nid in [train_ids[i] for i in tr_sub_idx]
                       if nid in node_idx_map]
    va_node_indices = [node_idx_map[nid] for nid in [train_ids[i] for i in va_sub_idx]
                       if nid in node_idx_map]

    y_tr = y_train[tr_sub_idx]
    y_va = y_train[va_sub_idx]

    tr_idx_t = torch.tensor(tr_node_indices, dtype=torch.long, device=device)
    va_idx_t = torch.tensor(va_node_indices, dtype=torch.long, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)

    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    in_dim = X_all_t.shape[1]
    model = DeepNDStyleGCN(
        in_dim=in_dim,
        n_experts=len(adj_norms),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_prauc = -1.0
    best_state = None
    no_improve = 0

    adj_norms = [adj.to(device) for adj in adj_norms]
    X_all_t = X_all_t.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        probs_all = model(adj_norms, X_all_t)
        probs_train = probs_all[tr_idx_t]
        loss = weighted_bce_from_probs(probs_train, y_tr_t, pos_weight)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            probs_all_val = model(adj_norms, X_all_t)
            val_scores = probs_all_val[va_idx_t].cpu().numpy()
        y_va_np = y_va.astype(int)

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
            print(f"  [GCN] Early stopping at epoch {epoch + 1}, best val PR-AUC={best_val_prauc:.4f}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model


def score_gcn(
    model: DeepNDStyleGCN,
    adj_norms: list[torch.Tensor],
    X_all_t: torch.Tensor,
    device: torch.device,
    return_parts: bool = False,
):
    model.eval()
    adj_norms = [adj.to(device) for adj in adj_norms]
    X_all_t = X_all_t.to(device)
    with torch.no_grad():
        if return_parts:
            probs, expert_probs, gate_weights = model(adj_norms, X_all_t, return_parts=True)
            return (
                probs.cpu().numpy(),
                expert_probs.cpu().numpy(),
                gate_weights.cpu().numpy(),
            )
        probs = model(adj_norms, X_all_t)
    return probs.cpu().numpy()


# ============================================================
# Main pipeline
# ============================================================

def run_gcn(
    labels_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    brainspan_df: pd.DataFrame,
    G: nx.Graph,
    ext_data_dir: Path,
    output_dir: Path,
    n_splits: int,
    random_state: int,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    patience: int,
    no_string_features: bool,
    expert_thresholds: list[int],
    label_noise_mode: str = "none",
    label_noise_rate: float = 0.0,
    label_budget_positive_fraction: float = 1.0,
    label_budget_neg_ratio: float = 0.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_df = labels_df.reset_index(drop=True)

    # For GCN: node universe = all nodes in STRING graph
    # Labels must be in meta_df AND G.nodes
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

    # Node list: all STRING graph nodes (for message passing); this is the transductive universe
    # We keep the node ordering stable across folds
    all_graph_nodes = sorted(G.nodes())
    node_idx_map = {node: i for i, node in enumerate(all_graph_nodes)}
    n_nodes = len(all_graph_nodes)
    print(f"[GCN] Graph has {n_nodes} nodes")

    # Build DeepND-style expert graphs. DeepND uses many co-expression networks;
    # here we keep the same input/output contract and derive multiple experts
    # from STRING confidence thresholds.
    print(f"[GCN] Building DeepND-style expert adjacencies: thresholds={expert_thresholds}")
    expert_graphs = build_string_threshold_graphs(
        ext_data_dir=ext_data_dir,
        thresholds=expert_thresholds,
        node_list=all_graph_nodes,
    )
    adj_norms = [build_normalized_adjacency(g, all_graph_nodes) for g in expert_graphs]
    expert_edge_counts = [int(g.number_of_edges()) for g in expert_graphs]
    print(f"[GCN] Expert edge counts: {expert_edge_counts}")

    # Build node feature matrix (meta + BrainSpan; NO STRING anchor features)
    use_brainspan = not no_string_features  # --no-string-features disables BrainSpan too (ablation)
    print(f"[GCN] Building node features (use_brainspan={use_brainspan})...")
    feat_matrix_raw = build_gcn_node_features(
        meta_df=meta_df,
        brainspan_df=brainspan_df,
        node_list=all_graph_nodes,
        use_brainspan=use_brainspan,
    )
    print(f"[GCN] Raw node feature matrix shape: {feat_matrix_raw.shape}")

    # For target_ids (meta universe), we need to map back scores to meta_df index
    target_ids = meta_df.index.astype(str).tolist()
    label_ids = set(labels_df["id"].astype(str))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X_ids = labels_df["id"].values
    y = labels_df["label"].values

    all_metrics: list[dict] = []
    full_scores_unlabeled: list[pd.DataFrame] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_ids, y), start=1):
        print(f"[GCN] Fold {fold_idx}/{n_splits}")
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

        # Standardize node features: fit scaler on training node features only
        train_node_indices = [node_idx_map[nid] for nid in train_ids if nid in node_idx_map]
        scaler = StandardScaler()
        feat_matrix_scaled = feat_matrix_raw.copy()
        if len(train_node_indices) > 0:
            scaler.fit(feat_matrix_raw[train_node_indices])
            feat_matrix_scaled = scaler.transform(feat_matrix_raw)

        X_all_t = torch.tensor(feat_matrix_scaled, dtype=torch.float32)
        in_dim = X_all_t.shape[1]

        model = train_gcn(
            adj_norms=adj_norms,
            X_all_t=X_all_t,
            node_idx_map=node_idx_map,
            train_ids=train_ids,
            y_train=y_train,
            device=device,
            hidden_dim=hidden_dim,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience,
            random_state=random_state + fold_idx,
        )

        # Score all nodes in the graph
        all_scores_np, expert_scores_np, gate_weights_np = score_gcn(
            model, adj_norms, X_all_t, device, return_parts=True
        )

        # Build scores Series indexed by all_graph_nodes
        graph_scores = pd.Series(all_scores_np, index=all_graph_nodes)

        # Map scores back to meta_df index (target universe)
        final_scores = graph_scores.reindex(target_ids).fillna(0.0)

        test_scores = final_scores.loc[test_df["id"]].to_numpy(dtype=float)
        test_metrics = evaluate_predictions(test_df["label"].to_numpy(dtype=int), test_scores)
        test_metrics["fold"] = fold_idx
        test_metrics["n_test"] = int(len(test_df))
        all_metrics.append(test_metrics)

        fold_pred_df = test_df.copy()
        fold_pred_df["forecASD"] = test_scores
        fold_pred_df["pred_label"] = (fold_pred_df["forecASD"] >= 0.5).astype(int)
        fold_pred_df.to_csv(fold_dir / "test_predictions.csv", index=False)

        component_df = pd.DataFrame(index=all_graph_nodes)
        component_df["forecASD"] = all_scores_np
        for i, threshold in enumerate(expert_thresholds):
            component_df[f"expert_score_thr_{threshold}"] = expert_scores_np[:, i]
            component_df[f"gate_weight_thr_{threshold}"] = gate_weights_np[:, i]
        component_df.reindex(target_ids).fillna(0.0).to_csv(
            fold_dir / "all_gene_component_scores.csv"
        )

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
                    "n_graph_nodes": n_nodes,
                    "in_dim": int(in_dim),
                    "use_brainspan": use_brainspan,
                    "architecture": "DeepND-ST-style MoE over STRING threshold GCN experts",
                    "expert_thresholds": expert_thresholds,
                    "expert_edge_counts": expert_edge_counts,
                    "expert_hidden_dim": hidden_dim,
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
        description="DeepND-ST-style GCN baseline with STRING threshold experts."
    )
    p.add_argument("--project-root", type=str, required=True, help="Directory containing ext_data/")
    p.add_argument("--labels-dir", type=str, default=None,
                   help="Label directory; defaults to project_root/forecasd_outputs")
    p.add_argument("--output-dir", type=str, default="gcn_outputs",
                   help="Output directory name under project root")
    p.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for CV splitting")
    p.add_argument("--string-mode", type=str, default="anchor", choices=["anchor", "graph"],
                   help="STRING feature mode (unused for GCN; graph topology used directly)")
    p.add_argument("--max-string-anchors", type=int, default=256,
                   help="Max anchors (unused for GCN)")
    p.add_argument("--force-rebuild-brainspan", action="store_true",
                   help="Rebuild cached BrainSpan matrix")
    p.add_argument("--force-rebuild-string", action="store_true",
                   help="Rebuild cached STRING graph")
    p.add_argument("--force-rebuild-graph-features", action="store_true",
                   help="(Unused for GCN 鈥?adjacency is always rebuilt from G)")
    # Neural network specific
    p.add_argument("--device", type=str, default="auto",
                   help="Device: auto / cpu / cuda (default=auto)")
    p.add_argument("--epochs", type=int, default=300, help="Max training epochs (default=300)")
    p.add_argument("--learning-rate", type=float, default=1e-2,
                   help="Adam learning rate (default=1e-2)")
    p.add_argument("--weight-decay", type=float, default=5e-4,
                   help="Adam weight decay (default=5e-4)")
    p.add_argument("--hidden-dim", type=int, default=4,
                   help="Expert GCN hidden dimension. DeepND uses 4 (default=4)")
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (default=0.5)")
    p.add_argument("--patience", type=int, default=30,
                   help="Early stopping patience on val PR-AUC (default=30)")
    p.add_argument("--no-string-features", action="store_true",
                   help="Ablation: use only meta features without BrainSpan for node features")
    p.add_argument("--expert-thresholds", type=str, default="400,700,900",
                   help="Comma-separated STRING score thresholds used as DeepND-style experts. "
                        "Default: 400,700,900")
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

    expert_thresholds = parse_expert_thresholds(args.expert_thresholds)
    print("[INFO] Running DeepND-ST-style GCN baseline (STRING threshold experts)...")
    run_gcn(
        labels_df=labels_df,
        meta_df=meta_df,
        brainspan_df=brainspan_df,
        G=G,
        ext_data_dir=ext_data_dir,
        output_dir=output_dir,
        n_splits=args.n_splits,
        random_state=args.random_state,
        label_noise_mode=args.label_noise_mode,
        label_noise_rate=args.label_noise_rate,
        label_budget_positive_fraction=args.label_budget_positive_fraction,
        label_budget_neg_ratio=args.label_budget_neg_ratio,
        device=device,
        epochs=args.epochs,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        patience=args.patience,
        no_string_features=args.no_string_features,
        expert_thresholds=expert_thresholds,
    )

    print(f"[DONE] Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

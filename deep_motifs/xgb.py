from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from statsmodels.nonparametric.smoothers_lowess import lowess
from xgboost import XGBClassifier


# ============================================================
# Noise injection utilities
# ============================================================

def _apply_feature_noise(
    X: np.ndarray,
    noise_type: str,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian or dropout noise applied to training features only."""
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
    """Flip flip_rate fraction of negative training labels to positive."""
    if flip_rate == 0.0:
        return y
    y = y.copy()
    neg_idx = np.where(y == 0)[0]
    n_flip = int(len(neg_idx) * flip_rate)
    if n_flip > 0:
        y[rng.choice(neg_idx, size=n_flip, replace=False)] = 1
    return y


# ============================================================
# Basic utilities
# ============================================================

def ensure_exists(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")


def normalize_value(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    return s


def save_pickle(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# Official input loading
# ============================================================

def load_composite_table(ext_data_dir: Path) -> pd.DataFrame:
    path = ext_data_dir / "composite_table.csv"
    ensure_exists(path, "composite_table.csv")
    meta = pd.read_csv(path, index_col=0)
    meta.index = meta.index.map(str)
    return meta


def build_tada_feature_matrix(
    tada_path: Path,
    jack_path: Path,
    target_proteins: set[str],
) -> pd.DataFrame:
    """
    Load tada_new.csv and merge with the protein-ID universe via jack_fu_gene_info(in).csv.

    Selected features (not present in composite_table.csv):
      - FDR_TADA_ASD, FDR_TADA_DD, FDR_TADA_NDD  (float, newer cohort FDRs)
      - ASD72, ASD185, Satterstrom102              (bool → int 0/1, ASD cohort flags)
      - DD309, DD477, NDD373, NDD664               (bool → int, DD/NDD flags)
      - SCZ10, SCZ244                              (bool → int, schizophrenia flags)
      - l10BF_ASD_PTV, l10BF_ASD_misB, l10BF_ASD_misA, l10BF_ASD_DEL, l10BF_ASD_DUP
        (log10 Bayes factors for variant types — richer signal than composite_table TADA_BF)

    Mapping: gene_id (ENSG) → ensembl_peptide_id (ENSP) via jack_fu_gene_info(in).csv.
    Fallback: hgnc_symbol → ensembl_peptide_id.
    Genes with multiple protein isoforms: numeric columns averaged, boolean columns maxed.
    Missing values filled with column median after joining.

    Returns DataFrame indexed by protein stable ID (ENSP), columns prefixed with 'tada_'.
    """
    fu = pd.read_csv(tada_path)
    jack = pd.read_csv(jack_path)

    bool_cols = [c for c in ["ASD72", "ASD185", "Satterstrom102", "DD309", "DD477",
                              "NDD373", "NDD664", "SCZ10", "SCZ244"] if c in fu.columns]
    float_cols = [c for c in ["FDR_TADA_ASD", "FDR_TADA_DD", "FDR_TADA_NDD",
                               "l10BF_ASD_PTV", "l10BF_ASD_misB", "l10BF_ASD_misA",
                               "l10BF_ASD_DEL", "l10BF_ASD_DUP",
                               "l10BF_DD_PTV", "l10BF_DD_misB", "l10BF_DD_misA"] if c in fu.columns]

    for col in bool_cols:
        fu[col] = (fu[col].astype(str).str.lower() == "true").astype(np.float32)
    for col in float_cols:
        fu[col] = pd.to_numeric(fu[col], errors="coerce").astype(np.float32)

    keep_cols = bool_cols + float_cols
    fu_clean = fu[["gene_id", "gene"] + keep_cols].copy() if "gene" in fu.columns else fu[["gene_id"] + keep_cols].copy()

    # Build gene_id → peptide and symbol → peptide maps
    jack = jack.dropna(subset=["ensembl_gene_id", "ensembl_peptide_id"])
    jack["ensembl_gene_id"] = jack["ensembl_gene_id"].astype(str).str.strip()
    jack["ensembl_peptide_id"] = jack["ensembl_peptide_id"].astype(str).str.strip()

    gene_to_peps: dict[str, list[str]] = {}
    sym_to_peps: dict[str, list[str]] = {}
    for _, row in jack.iterrows():
        gid = str(row["ensembl_gene_id"]).strip()
        pid = str(row["ensembl_peptide_id"]).strip()
        if gid and pid and pid != "nan":
            gene_to_peps.setdefault(gid, []).append(pid)
        if "hgnc_symbol" in jack.columns:
            sym = str(row.get("hgnc_symbol", "")).strip().upper()
            if sym and sym != "NAN" and pid and pid != "nan":
                sym_to_peps.setdefault(sym, []).append(pid)

    # Expand tada rows to protein IDs
    records: list[dict] = []
    for _, row in fu_clean.iterrows():
        gid = str(row.get("gene_id", "")).strip()
        pids = gene_to_peps.get(gid, [])
        if not pids and "gene" in fu_clean.columns:
            sym = str(row.get("gene", "")).strip().upper()
            pids = sym_to_peps.get(sym, [])
        for pid in pids:
            if pid in target_proteins:
                rec = {"protein_id": pid}
                for col in keep_cols:
                    rec[col] = float(row[col]) if not pd.isna(row[col]) else float("nan")
                records.append(rec)

    if not records:
        print("[WARN] build_tada_feature_matrix: no proteins mapped — returning empty frame")
        return pd.DataFrame(index=sorted(target_proteins))

    df = pd.DataFrame(records).set_index("protein_id")
    # Aggregate duplicates: mean for floats, max for boolean flags
    agg = {col: "max" if col in bool_cols else "mean" for col in keep_cols}
    df = df.groupby(df.index).agg(agg)

    # Reindex to full target set, fill missing with column median
    df = df.reindex(sorted(target_proteins))
    df = df.fillna(df.median(numeric_only=True)).fillna(0.0)

    # Prefix columns to avoid name collisions with composite_table
    df.columns = ["tada_" + c for c in df.columns]
    print(f"[INFO] build_tada_feature_matrix: {df.shape[1]} features for {(df != 0).any(axis=1).sum()} / {len(df)} proteins")
    return df


def augment_composite_with_tada(
    meta_df: pd.DataFrame,
    tada_path: Path,
    jack_path: Path,
) -> pd.DataFrame:
    """
    Append tada_new.csv features to composite_table (meta_df) and return the merged frame.
    Proteins not covered by tada_new get median-imputed values.
    Called from main() in all training scripts to always augment meta features.
    """
    tada_df = build_tada_feature_matrix(
        tada_path=tada_path,
        jack_path=jack_path,
        target_proteins=set(meta_df.index.astype(str)),
    )
    tada_df = tada_df.reindex(meta_df.index)
    return pd.concat([meta_df, tada_df], axis=1)


def load_labels(labels_dir: Path) -> pd.DataFrame:
    all_labels = labels_dir / "all_labels_used.csv"
    labels = labels_dir / "labels_used.csv"
    if all_labels.exists():
        df = pd.read_csv(all_labels)
    elif labels.exists():
        df = pd.read_csv(labels)
    else:
        raise FileNotFoundError(f"No labels found in {labels_dir} (expected all_labels_used.csv or labels_used.csv)")
    if "id" not in df.columns or "label" not in df.columns:
        raise ValueError("Labels file must contain columns: id, label")
    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    return df


# ============================================================
# BrainSpan feature construction (same logic as forecasd.py)
# ============================================================

def convert_age_to_weeks(age_str: str) -> float:
    parts = str(age_str).strip().split()
    if len(parts) < 2:
        raise ValueError(f"Unexpected age format: {age_str}")
    value = float(parts[0])
    unit = parts[1]
    if unit == "pcw":
        return value
    if unit == "mos":
        return value * 4.33 + 38.0
    if unit == "yrs":
        return value * 52.0 + 38.0
    raise ValueError(f"Unexpected age unit: {age_str}")


def lowess_interpolate(y: np.ndarray, x_age: np.ndarray, n_points: int = 50) -> np.ndarray:
    x_log = np.log(x_age.astype(float))
    smoothed = lowess(endog=y.astype(float), exog=x_log, frac=1 / 3, return_sorted=True)
    target_x = np.linspace(2.0, np.log(2118.0), num=n_points)
    out = np.interp(target_x, smoothed[:, 0], smoothed[:, 1], left=smoothed[0, 1], right=smoothed[-1, 1])
    return out


def build_brainspan_matrix(
    ext_data_dir: Path,
    target_proteins: set[str],
    force_rebuild: bool = False,
) -> pd.DataFrame:
    cache_path = ext_data_dir.parent / "cache" / "brainspan_matrix.pkl"
    if cache_path.exists() and not force_rebuild:
        df = load_pickle(cache_path)
        return df.loc[df.index.intersection(pd.Index(sorted(target_proteins)))]

    m_path = ext_data_dir / "brainspan" / "expression_matrix.csv"
    ann_path = ext_data_dir / "brainspan" / "rows_metadata.csv"
    fac_path = ext_data_dir / "brainspan" / "columns_metadata.csv"
    egmap_path = ext_data_dir / "entrez_ids" / "entrezgene2symbol.csv"
    e2e_path = ext_data_dir / "entrez_ids" / "entrez_gene_id.vs.string.v10.28042015.tsv"
    missing_path = ext_data_dir / "brainspan" / "brainspan_missing_ids.txt"

    for p, name in [
        (m_path, "BrainSpan expression_matrix.csv"),
        (ann_path, "BrainSpan rows_metadata.csv"),
        (fac_path, "BrainSpan columns_metadata.csv"),
        (egmap_path, "entrezgene2symbol.csv"),
        (e2e_path, "entrez_gene_id.vs.string.v10.28042015.tsv"),
        (missing_path, "brainspan_missing_ids.txt"),
    ]:
        ensure_exists(p, name)

    m = pd.read_csv(m_path, header=None)
    ann = pd.read_csv(ann_path)
    fac = pd.read_csv(fac_path)
    M = m.iloc[:, 1:].to_numpy(dtype=float)

    eg_map = pd.read_csv(egmap_path)
    symbol_to_entrez = {
        normalize_value(row["symbol"]): normalize_value(row["entrez"])
        for _, row in eg_map.iterrows()
    }

    ann = ann.copy()
    if "entrez_id" not in ann.columns or "gene_symbol" not in ann.columns:
        raise ValueError("rows_metadata.csv must contain entrez_id and gene_symbol")

    ann["entrez_id"] = ann["entrez_id"].map(normalize_value)
    ann["gene_symbol"] = ann["gene_symbol"].map(normalize_value)

    missing_mask = ann["entrez_id"] == ""
    ann.loc[missing_mask, "entrez_id"] = ann.loc[missing_mask, "gene_symbol"].map(symbol_to_entrez).fillna("")

    row_entrez = ann["entrez_id"].tolist()

    age_wk = fac["age"].map(convert_age_to_weeks).to_numpy(dtype=float)
    structure = fac["structure_acronym"].astype(str)
    keep_structures = structure.value_counts()
    keep_structures = keep_structures[keep_structures > 20].index.tolist()

    expr_by_region: list[np.ndarray] = []

    for region in keep_structures:
        mask = (structure == region).to_numpy()
        X = M[:, mask]
        x_age = age_wk[mask]
        region_out = np.zeros((X.shape[0], 50), dtype=np.float32)
        for i in range(X.shape[0]):
            region_out[i, :] = lowess_interpolate(X[i, :], x_age)
        expr_by_region.append(region_out)

    n_regions = len(expr_by_region)
    if n_regions == 0:
        raise ValueError("No BrainSpan structures passed the >20 samples filter")

    gene_mats = np.stack(expr_by_region, axis=1)
    gene_mats_flat = gene_mats.reshape(gene_mats.shape[0], -1).astype(np.float32)
    mean = gene_mats_flat.mean(axis=1, keepdims=True)
    std = gene_mats_flat.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    gene_mats_flat = (gene_mats_flat - mean) / std

    e2e = pd.read_csv(e2e_path, sep="\t", header=None, names=["entrez", "protein"], dtype=str)
    e2e["protein"] = e2e["protein"].astype(str).str.replace("9606.", "", regex=False)
    entrez_to_protein = {
        normalize_value(row["entrez"]): normalize_value(row["protein"])
        for _, row in e2e.iterrows()
    }

    missing_df = pd.read_csv(missing_path, sep="\t")
    if "NCBI.gene.ID" in missing_df.columns and "Protein.stable.ID" in missing_df.columns:
        for _, row in missing_df.iterrows():
            entrez_to_protein[normalize_value(row["NCBI.gene.ID"])] = normalize_value(row["Protein.stable.ID"])

    proteins = [entrez_to_protein.get(normalize_value(e), "") for e in row_entrez]
    valid = [i for i, p in enumerate(proteins) if p and p in target_proteins]

    df = pd.DataFrame(
        gene_mats_flat[valid, :],
        index=[proteins[i] for i in valid],
        columns=[f"bs_{j+1}" for j in range(gene_mats_flat.shape[1])],
    )
    df = df.fillna(df.median(numeric_only=True))
    df = df[~df.index.duplicated(keep="first")]

    save_pickle(df, cache_path)
    return df


# ============================================================
# STRING graph and features
# ============================================================

def build_string_graph(ext_data_dir: Path, force_rebuild: bool = False) -> nx.Graph:
    cache_path = ext_data_dir.parent / "cache" / "string_graph.pkl"
    if cache_path.exists() and not force_rebuild:
        return load_pickle(cache_path)

    string_path = ext_data_dir / "9606.protein.links.v10.txt.gz"
    ensure_exists(string_path, "9606.protein.links.v10.txt.gz")

    G = nx.Graph()
    with gzip.open(string_path, "rt") as f:
        header = next(f)
        for line in f:
            a, b, score = line.strip().split()
            if int(score) <= 400:
                continue
            a = a.replace("9606.", "")
            b = b.replace("9606.", "")
            G.add_edge(a, b)

    save_pickle(G, cache_path)
    return G


def build_fold_string_feature_matrix(
    G: nx.Graph,
    target_ids: list[str],
    anchor_ids: list[str],
    max_anchors: int = 256,
) -> pd.DataFrame:
    anchors = [a for a in anchor_ids if a in G]
    if len(anchors) == 0:
        raise ValueError("No training anchors are present in STRING graph")

    anchors = sorted(set(anchors))[:max_anchors]
    unreachable = 9999
    data = {anchor: np.full(len(target_ids), unreachable, dtype=np.int32) for anchor in anchors}
    index_lookup = {pid: i for i, pid in enumerate(target_ids)}

    for anchor in anchors:
        try:
            lengths = nx.single_source_shortest_path_length(G, anchor)
        except (KeyError, nx.NetworkXError, nx.NodeNotFound):
            # Skip anchors whose graph adjacency is inconsistent (e.g. corrupted cache)
            continue
        for node, dist in lengths.items():
            j = index_lookup.get(node)
            if j is not None:
                data[anchor][j] = int(dist)

    df = pd.DataFrame(data, index=target_ids)
    return df


def compute_graph_features(
    G: nx.Graph,
    target_ids: list[str],
    cache_path: Path,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    if cache_path.exists() and not force_rebuild:
        df = load_pickle(cache_path)
        return df.loc[df.index.intersection(pd.Index(target_ids))]

    degree = dict(G.degree())
    clustering = nx.clustering(G)
    pagerank = nx.pagerank(G)
    core_number = nx.core_number(G)

    component_size: dict[str, int] = {}
    for comp in nx.connected_components(G):
        size = len(comp)
        for node in comp:
            component_size[node] = size

    df = pd.DataFrame(
        {
            "string_degree_log": {k: float(np.log1p(v)) for k, v in degree.items()},
            "string_clustering": clustering,
            "string_pagerank": pagerank,
            "string_kcore": core_number,
            "string_component_size": component_size,
        }
    )
    df.index = df.index.astype(str)
    save_pickle(df, cache_path)
    return df.loc[df.index.intersection(pd.Index(target_ids))]


# ============================================================
# Model helpers and metrics
# ============================================================

def fit_xgb_and_score(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_all: pd.DataFrame,
    n_estimators: int,
    random_state: int,
) -> tuple[object, pd.Series]:
    y_unique = np.unique(y_train)
    if y_unique.size < 2:
        constant = int(y_unique[0]) if y_unique.size == 1 else 0
        clf = DummyClassifier(strategy="constant", constant=constant)
        clf.fit(X_train, y_train)
        scores = np.full(X_all.shape[0], float(constant), dtype=float)
        return clf, pd.Series(scores, index=X_all.index)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        learning_rate=0.05,
        max_depth=4,
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


def coerce_numeric_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert to numeric matrix and ensure there are no NaN/inf values.

    Some composite_table columns may be non-numeric; they become all-NaN after coercion.
    To keep training stable across estimators, we drop all-NaN columns, impute
    remaining NaN with column medians, then fall back to 0 for residual missingness.
    """
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(axis=1, how="all")
    out = out.fillna(out.median(numeric_only=True))
    out = out.fillna(0.0)
    return out


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ks: tuple[int, ...] = (10, 20, 50),
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = int(y_true.size)
    if n == 0:
        return {f"{metric}@{k}": float("nan") for k in ks for metric in ["precision", "recall", "lift", "ndcg"]}

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    n_pos = int((y_true == 1).sum())
    base_rate = (n_pos / n) if n > 0 else 0.0

    out: dict[str, float] = {}
    for k in ks:
        k_eff = min(int(k), n)
        if k_eff <= 0:
            precision = float("nan")
            recall = float("nan")
            lift = float("nan")
            ndcg = float("nan")
        else:
            top = y_sorted[:k_eff]
            tp = int((top == 1).sum())
            precision = tp / k_eff
            recall = (tp / n_pos) if n_pos > 0 else float("nan")
            lift = (precision / base_rate) if base_rate > 0 else float("nan")
            if n_pos > 0:
                denom = np.log2(np.arange(2, k_eff + 2))
                dcg = float(np.sum(top / denom))
                ideal_k = min(n_pos, k_eff)
                idcg = float(np.sum(np.ones(ideal_k) / np.log2(np.arange(2, ideal_k + 2))))
                ndcg = (dcg / idcg) if idcg > 0 else float("nan")
            else:
                ndcg = float("nan")

        out[f"precision@{k}"] = float(precision)
        out[f"recall@{k}"] = float(recall)
        out[f"lift@{k}"] = float(lift)
        out[f"ndcg@{k}"] = float(ndcg)

    return out


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    out.update(compute_ranking_metrics(y_true, y_score))
    return out


# ============================================================
# RF pipeline
# ============================================================

def build_feature_matrix(
    meta_df: pd.DataFrame,
    brainspan_df: pd.DataFrame,
    string_df: pd.DataFrame,
) -> pd.DataFrame:
    # Match forecasd.py final-model meta predictors:
    # after adding two score columns, it drops columns 3:9 (1-based),
    # which corresponds to dropping the first 7 columns of the original meta_df.
    meta_cols = meta_df.columns.tolist()
    keep_meta_cols = [c for i, c in enumerate(meta_cols, start=1) if i > 7]
    work_meta = meta_df[keep_meta_cols] if keep_meta_cols else meta_df.copy()
    meta_num = coerce_numeric_and_impute(work_meta)

    bs_median = brainspan_df.median(numeric_only=True)
    bs_all = brainspan_df.reindex(meta_df.index).fillna(bs_median)
    bs_all = bs_all.replace([np.inf, -np.inf], np.nan)
    bs_all = bs_all.fillna(bs_all.median(numeric_only=True)).fillna(0.0)

    X_all = pd.concat([meta_num, bs_all, string_df], axis=1)
    X_all = coerce_numeric_and_impute(X_all)
    return X_all


def run_xgb(
    labels_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    brainspan_df: pd.DataFrame,
    G: nx.Graph,
    output_dir: Path,
    string_mode: str,
    max_string_anchors: int,
    n_splits: int,
    random_state: int,
    force_rebuild_graph_features: bool,
    noise_type: str = "none",
    noise_level: float = 0.0,
    label_flip_rate: float = 0.0,
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
        cache_path = output_dir.parent / "cache" / "string_graph_features.pkl"
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
        train_df.to_csv(fold_dir / "train_labels.tsv", sep="\t", index=False)
        test_df.to_csv(fold_dir / "test_labels.tsv", sep="\t", index=False)

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

        X_train = X_all.loc[train_ids]
        if noise_type != "none" and noise_level > 0.0:
            _arr = _apply_feature_noise(X_train.to_numpy(dtype=np.float32), noise_type, noise_level, rng)
            X_train = pd.DataFrame(_arr, index=X_train.index, columns=X_train.columns)
        _, final_scores = fit_xgb_and_score(
            X_train=X_train,
            y_train=y_train,
            X_all=X_all,
            n_estimators=600,
            random_state=43775,
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
            "ensembl_string": final_scores.index,
            "forecASD": final_scores.values,
        })
        full_scores_df = full_scores_df[~full_scores_df["ensembl_string"].isin(label_ids)]
        full_scores_df.to_csv(fold_dir / "full_scores.csv", index=False)
        full_scores_unlabeled.append(full_scores_df.set_index("ensembl_string"))

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
        summary_scores = summary_scores.reset_index().rename(columns={"index": "ensembl_string"})
        summary_scores.to_csv(output_dir / "full_scores_summary.csv", index=False)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single XGBoost baseline using composite + BrainSpan + STRING features."
    )
    p.add_argument("--project-root", type=str, required=True, help="Directory containing ext_data/")
    p.add_argument("--labels-dir", type=str, default=None, help="Label directory; defaults to project_root/forecasd_outputs")
    p.add_argument("--output-dir", type=str, default="xgb_outputs", help="Output directory name under project root")
    p.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for CV splitting")
    p.add_argument("--string-mode", type=str, default="anchor", choices=["anchor", "graph"], help="STRING feature mode")
    p.add_argument("--max-string-anchors", type=int, default=256, help="Max anchors for string-mode=anchor")
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

    print(f"[INFO] Running XGB baseline (string_mode={args.string_mode})...")
    run_xgb(
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
        noise_type=args.noise_type,
        noise_level=args.noise_level,
        label_flip_rate=args.label_flip_rate,
    )

    print(f"[DONE] Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

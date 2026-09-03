from __future__ import annotations

import argparse
import gzip
import json
import pickle
import zipfile
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from experiments.label_noise_utils import (
    add_label_budget_args,
    add_label_noise_args,
    apply_training_label_budget,
    apply_training_label_perturbation,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
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

from experiments.gene_id_utils import (
    aggregate_table_to_gene_level,
    build_gene_string_graph,
    load_gene_mappings,
    map_identifiers_to_genes,
)


# ============================================================
# Basic utilities
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


def ensure_exists(path: Path, desc: str) -> None:
    """Ensure a file or directory exists."""
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")


def normalize_value(x: object) -> str:
    """Normalize identifiers into stripped strings."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    return s




def save_pickle(obj: object, path: Path) -> None:
    """Save Python object as pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> object:
    """Load Python object from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# Identifier mapping
# ============================================================

def build_mapping_dicts(sfari_gene_ids_path: Path) -> dict[str, dict[str, set[str]]]:
    """
    Build mappings from multiple gene identifier types to ENSG gene ID.

    Required columns in sfari_gene_ids.txt:
    - Gene stable ID
    - Gene name
    - NCBI gene ID
    - Protein stable ID
    """
    return load_gene_mappings(sfari_gene_ids_path.parent)


def map_identifiers_to_gene_ids(
    identifiers: Iterable[object],
    mapping: dict[str, dict[str, set[str]]],
) -> set[str]:
    """Map arbitrary identifiers to ENSG gene IDs."""
    return map_identifiers_to_genes(list(identifiers), mapping)


def map_identifiers_to_proteins(
    identifiers: Iterable[object],
    mapping: dict[str, dict[str, set[str]]],
) -> set[str]:
    """Backward-compatible alias; the pipeline now returns ENSG gene IDs."""
    return map_identifiers_to_gene_ids(identifiers, mapping)


# ============================================================
# Label generation using new logic
# ============================================================

def extract_positive_ids_from_sfari(sfari_new_path: Path, mapping: dict[str, dict[str, set[str]]]) -> set[str]:
    """
    Extract positive samples from the latest SFARI file.

    Fixed columns in SFARI_new.csv:
    - ensembl-id
    - gene-score

    Positive definition:
    gene-score in {{1, 2}}
    """
    sfari = pd.read_csv(sfari_new_path)

    # convert to numeric first; anything that cannot be converted becomes NaN
    sfari["gene-score-num"] = pd.to_numeric(sfari["gene-score"], errors="coerce")

    # keep only 1 and 2
    pos = sfari[sfari["gene-score-num"].isin([1, 2])].copy()

    # map using ensembl-id
    pos_ids = pos["ensembl-id"].dropna().astype(str).str.strip().tolist()

    pos_ids = map_identifiers_to_gene_ids(pos_ids, mapping)

    return pos_ids





def create_label_table(
    pos_ids: set[str],
    neg_ids: set[str],
    valid_ids: set[str],
    neg_ratio: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build the final label table and restrict labels to IDs present in official inputs.

    Rule:
    - positives have priority
    - overlap is removed from negatives
    - optionally downsample negatives to neg_ratio * positives
    """
    pos_ids = pos_ids & valid_ids
    neg_ids = neg_ids & valid_ids

    overlap = pos_ids & neg_ids
    if overlap:
        print(f"[WARN] Positive/negative overlap detected: {len(overlap)} IDs")
        print(f"[WARN] Removing overlap from negatives. Example: {sorted(list(overlap))[:10]}")
        neg_ids = neg_ids - overlap

    if neg_ratio is not None and len(pos_ids) > 0:
        target = int(len(pos_ids) * float(neg_ratio))
        if len(neg_ids) > target:
            rng = np.random.default_rng(random_state)
            neg_ids = set(rng.choice(sorted(neg_ids), size=target, replace=False).tolist())

    pos_df = pd.DataFrame({"id": sorted(pos_ids), "label": 1})
    neg_df = pd.DataFrame({"id": sorted(neg_ids), "label": 0})
    labels = pd.concat([pos_df, neg_df], ignore_index=True)

    if len(labels) == 0:
        raise ValueError("No labels left after filtering valid IDs and overlap removal.")

    return labels


def extract_sfari_reference_exclusion_ids(
    sfari_new_path: Path,
    mapping: dict[str, dict[str, set[str]]],
    report_min: int = 3,
    eagle_min: float = 1.0,
    exclude_syndromic: bool = True,
    exclude_all_listed: bool = True,
) -> tuple[set[str], pd.DataFrame]:
    """
    Build an exclusion set from SFARI reference fields, then use the remaining universe
    as the random negative sampling pool.

    By default, exclude genes if any of the following are true:
    - gene-score in {1, 2, 3}
    - syndromic == 1
    - number-of-reports >= report_min
    - eagle >= eagle_min

    If exclude_all_listed=True (default), additionally exclude every gene that appears
    anywhere in SFARI_new.csv regardless of score, syndromic or eagle values.
    This ensures no SFARI-mentioned gene can end up as a labeled negative.
    """
    sfari = pd.read_csv(sfari_new_path).copy()
    for col in ["gene-score", "syndromic", "eagle", "number-of-reports", "ensembl-id", "gene-symbol"]:
        if col not in sfari.columns:
            raise ValueError(f"SFARI_new.csv missing required column: {col}")

    sfari["gene-score-num"] = pd.to_numeric(sfari["gene-score"], errors="coerce")
    sfari["syndromic-num"] = pd.to_numeric(sfari["syndromic"], errors="coerce").fillna(0)
    sfari["eagle-num"] = pd.to_numeric(sfari["eagle"], errors="coerce")
    sfari["reports-num"] = pd.to_numeric(sfari["number-of-reports"], errors="coerce")

    masks: list[pd.Series] = []
    reason_cols: list[str] = []

    score_mask = sfari["gene-score-num"].isin([1, 2, 3])
    masks.append(score_mask)
    reason_cols.append("exclude_gene_score_le_3")

    if exclude_syndromic:
        synd_mask = sfari["syndromic-num"] >= 1
        masks.append(synd_mask)
        reason_cols.append("exclude_syndromic")

    report_mask = sfari["reports-num"] >= int(report_min)
    masks.append(report_mask)
    reason_cols.append("exclude_report_min")

    eagle_mask = sfari["eagle-num"] >= float(eagle_min)
    masks.append(eagle_mask)
    reason_cols.append("exclude_eagle_min")

    risk_mask = pd.Series(False, index=sfari.index)
    for m in masks:
        risk_mask = risk_mask | m.fillna(False)
    risk_df = sfari[risk_mask].copy()

    risk_df["exclude_gene_score_le_3"] = score_mask.loc[risk_df.index].astype(int)
    if exclude_syndromic:
        risk_df["exclude_syndromic"] = synd_mask.loc[risk_df.index].astype(int)
    risk_df["exclude_report_min"] = report_mask.loc[risk_df.index].astype(int)
    risk_df["exclude_eagle_min"] = eagle_mask.loc[risk_df.index].astype(int)

    gene_ids: set[str] = set()
    mapped_ids: list[list[str]] = []
    for _, row in risk_df.iterrows():
        ids = []
        ensembl_id = normalize_value(row["ensembl-id"])
        gene_symbol = normalize_value(row["gene-symbol"])
        if ensembl_id:
            ids.append(ensembl_id)
        if gene_symbol:
            ids.append(gene_symbol)
        mapped = sorted(map_identifiers_to_gene_ids(ids, mapping))
        gene_ids.update(mapped)
        mapped_ids.append(mapped)

    risk_df["mapped_gene_ids"] = [";".join(x) for x in mapped_ids]

    # Additionally exclude every gene that appears anywhere in SFARI_new.csv.
    # Any SFARI-mentioned gene is a potential ASD candidate regardless of score.
    if exclude_all_listed:
        for _, row in sfari.iterrows():
            ids = []
            ensembl_id = normalize_value(row["ensembl-id"])
            gene_symbol = normalize_value(row["gene-symbol"])
            if ensembl_id:
                ids.append(ensembl_id)
            if gene_symbol:
                ids.append(gene_symbol)
            gene_ids.update(map_identifiers_to_gene_ids(ids, mapping))
        print(f"[INFO] SFARI exclusion (all listed): {len(gene_ids)} gene IDs excluded")

    return gene_ids, risk_df


def build_string_proximity_tiers(
    G: nx.Graph,
    pos_ids: set[str],
    candidate_pool: set[str],
    max_hops: int = 2,
) -> tuple[set[str], set[str]]:
    """
    Multi-source BFS from all positive genes in STRING graph.

    Returns:
        tier_a: candidates reachable within max_hops of any positive gene
                (network-adjacent → biologically ambiguous, harder negatives)
        tier_b: candidates beyond max_hops
                (network-distant → clearer negatives)
    """
    pos_in_graph = set(pos_ids) & set(G.nodes)
    if not pos_in_graph:
        return set(), set(candidate_pool)

    visited = set(pos_in_graph)
    frontier = set(pos_in_graph)
    for _ in range(max_hops):
        next_frontier = set()
        for node in frontier:
            for nb in G.neighbors(node):
                if nb not in visited:
                    next_frontier.add(nb)
                    visited.add(nb)
        frontier = next_frontier

    tier_a = set(candidate_pool) & (visited - pos_in_graph)
    tier_b = set(candidate_pool) - visited
    return tier_a, tier_b


def select_negatives_random_from_reference_filtered(
    pos_ids: set[str],
    reference_exclude_ids: set[str],
    valid_ids: set[str],
    target_neg_ratio: float,
    random_state: int,
    G: nx.Graph | None = None,
    tier_a_ratio: float = 0.0,
    tier_a_max_hops: int = 2,
) -> tuple[pd.DataFrame, set[str], dict[str, int | float]]:
    """
    Sample negatives from the valid universe after excluding positives and
    SFARI-reference high-risk genes.

    When G is provided and tier_a_ratio > 0, uses STRING-aware stratified sampling:
      Tier A (tier_a_ratio of target): STRING network neighbours of positives
              within tier_a_max_hops.  These are biologically ambiguous genes
              that interact with known ASD risk genes — harder, noisier negatives
              that reduce XGBoost's trivial distance advantage.
      Tier B (remainder): genes beyond tier_a_max_hops — clearer negatives.

    Falls back to pure random sampling when G is None or tier_a_ratio == 0.
    """
    pos = set(pos_ids) & set(valid_ids)
    exclude = (set(reference_exclude_ids) & set(valid_ids)) - pos
    candidate_pool = set(valid_ids) - pos - exclude
    if not pos:
        raise ValueError("No positives left after filtering valid IDs.")
    if not candidate_pool:
        raise ValueError("No candidate negatives left after SFARI-reference exclusion.")

    target_neg = max(1, int(len(pos) * float(max(target_neg_ratio, 0.0))))
    rng = np.random.default_rng(random_state)

    if G is not None and tier_a_ratio > 0.0:
        tier_a, tier_b = build_string_proximity_tiers(G, pos, candidate_pool, tier_a_max_hops)
        n_want_a = int(round(target_neg * float(tier_a_ratio)))
        n_tier_a = min(n_want_a, len(tier_a))
        n_tier_b = min(target_neg - n_tier_a, len(tier_b))
        sel_a = (
            set(rng.choice(sorted(tier_a), size=n_tier_a, replace=False).tolist())
            if n_tier_a > 0 else set()
        )
        sel_b = (
            set(rng.choice(sorted(tier_b), size=n_tier_b, replace=False).tolist())
            if n_tier_b > 0 else set()
        )
        selected_neg = sel_a | sel_b
        # Ensure we reach target_neg whenever enough candidates exist.
        # This avoids underfilling when one tier (commonly tier_b) is small.
        n_missing = max(target_neg - len(selected_neg), 0)
        if n_missing > 0:
            extra_pool = sorted(candidate_pool - selected_neg)
            extra_n = min(n_missing, len(extra_pool))
            extra_sel = (
                set(rng.choice(extra_pool, size=extra_n, replace=False).tolist())
                if extra_n > 0 else set()
            )
            sel_a.update(extra_sel & tier_a)
            sel_b.update(extra_sel & tier_b)
            selected_neg = sel_a | sel_b
            n_tier_a = len(sel_a)
            n_tier_b = len(sel_b)

        print(
            f"[INFO] STRING-aware sampling: "
            f"tier_a_available={len(tier_a)} selected={n_tier_a} | "
            f"tier_b_available={len(tier_b)} selected={n_tier_b} | "
            f"total_neg={len(selected_neg)}"
        )
    else:
        final_count = min(target_neg, len(candidate_pool))
        selected_neg = (
            set(rng.choice(sorted(candidate_pool), size=final_count, replace=False).tolist())
            if len(candidate_pool) > final_count else set(candidate_pool)
        )
        tier_a, tier_b = set(), set()
        n_tier_a, n_tier_b = 0, len(selected_neg)

    labels_df = create_label_table(
        pos_ids=pos,
        neg_ids=selected_neg,
        valid_ids=valid_ids,
        neg_ratio=None,
        random_state=random_state,
    )
    report = {
        "n_pos": int(len(pos)),
        "target_neg": int(target_neg),
        "n_reference_excluded": int(len(exclude)),
        "n_candidate_pool": int(len(candidate_pool)),
        "n_tier_a_available": int(len(tier_a)),
        "n_tier_b_available": int(len(tier_b)),
        "n_selected_tier_a": int(n_tier_a),
        "n_selected_tier_b": int(n_tier_b),
        "n_selected_neg_total": int(len(selected_neg)),
        "achieved_neg_ratio": float(len(selected_neg) / len(pos)) if len(pos) > 0 else float("nan"),
    }
    return labels_df, selected_neg, report


def load_deepnd_negative_ids(path: Path) -> set[str]:
    """Load DeepND negative IDs and map them to ENSG gene IDs."""
    ensure_exists(path, "DeepND mapped negative ID file")
    raw_ids = {
        normalize_value(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if normalize_value(line)
    }
    mappings = load_gene_mappings(path.parent)
    gene_ids = map_identifiers_to_genes(sorted(raw_ids), mappings)
    # If the file has already been converted to ENSG, preserve those entries.
    gene_ids.update({x for x in raw_ids if x.startswith("ENSG")})
    return gene_ids


def select_negatives_deepnd_then_reference_fill(
    pos_ids: set[str],
    deepnd_neg_ids: set[str],
    reference_exclude_ids: set[str],
    valid_ids: set[str],
    target_neg_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, set[str], dict[str, int | float | str]]:
    """
    Negative selection using published DeepND ASD negatives first, then fill.

    Steps:
    1. Keep DeepND negative gene IDs that are in our valid universe.
    2. Remove any overlap with our current positives.
    3. If fewer than target_neg_ratio * positives remain, fill the gap with the
       original SFARI-reference-filtered random candidate pool.

    DeepND negatives are retained even if they would have been excluded by the
    SFARI-reference filter, because this strategy is meant to use the published
    negative set directly after positive-overlap removal.
    """
    pos = set(pos_ids) & set(valid_ids)
    if not pos:
        raise ValueError("No positives left after filtering valid IDs.")

    target_neg = max(1, int(len(pos) * float(max(target_neg_ratio, 0.0))))
    deepnd_raw = set(deepnd_neg_ids)
    deepnd_in_valid = deepnd_raw & set(valid_ids)
    deepnd_pos_overlap = deepnd_in_valid & pos
    deepnd_clean = deepnd_in_valid - pos

    selected_neg = set(sorted(deepnd_clean)[:target_neg])
    n_deepnd_selected = len(selected_neg)
    n_fill_needed = max(target_neg - len(selected_neg), 0)

    reference_exclude = (set(reference_exclude_ids) & set(valid_ids)) - pos
    fill_pool = set(valid_ids) - pos - reference_exclude - selected_neg
    rng = np.random.default_rng(random_state)
    fill_selected: set[str] = set()
    if n_fill_needed > 0:
        if not fill_pool:
            raise ValueError("No candidate negatives left for DeepND fill step.")
        fill_n = min(n_fill_needed, len(fill_pool))
        fill_selected = set(
            rng.choice(sorted(fill_pool), size=fill_n, replace=False).tolist()
        )
        selected_neg.update(fill_selected)

    labels_df = create_label_table(
        pos_ids=pos,
        neg_ids=selected_neg,
        valid_ids=valid_ids,
        neg_ratio=None,
        random_state=random_state,
    )
    report = {
        "strategy": "deepnd_reference_fill",
        "n_pos": int(len(pos)),
        "target_neg": int(target_neg),
        "n_deepnd_raw": int(len(deepnd_raw)),
        "n_deepnd_in_valid_universe": int(len(deepnd_in_valid)),
        "n_deepnd_overlap_with_current_positives": int(len(deepnd_pos_overlap)),
        "n_deepnd_clean_available": int(len(deepnd_clean)),
        "n_deepnd_selected": int(n_deepnd_selected),
        "n_reference_excluded_for_fill": int(len(reference_exclude)),
        "n_fill_pool": int(len(fill_pool)),
        "n_fill_selected": int(len(fill_selected)),
        "n_selected_neg_total": int(len(selected_neg)),
        "achieved_neg_ratio": float(len(selected_neg) / len(pos)) if len(pos) > 0 else float("nan"),
    }
    return labels_df, selected_neg, report


def save_label_outputs(labels_df: pd.DataFrame, output_dir: Path) -> None:
    """Persist labels for reuse in downstream experiments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(output_dir / "labels_used.csv", index=False)
    pos_ids = labels_df.loc[labels_df["label"] == 1, "id"].tolist()
    neg_ids = labels_df.loc[labels_df["label"] == 0, "id"].tolist()
    (output_dir / "pos_ids.txt").write_text("\n".join(pos_ids), encoding="utf-8")
    (output_dir / "neg_ids.txt").write_text("\n".join(neg_ids), encoding="utf-8")

# ============================================================
# Official input loading
# ============================================================

def load_composite_table(ext_data_dir: Path) -> pd.DataFrame:
    """
    Load official composite_table.csv.

    The official R code uses row.names=1, so we do the same by setting index_col=0.
    """
    path = ext_data_dir / "composite_table.csv"
    ensure_exists(path, "composite_table.csv")
    meta = pd.read_csv(path, index_col=0)
    meta.index = meta.index.map(str)
    meta_gene = aggregate_table_to_gene_level(meta, ext_data_dir)
    print(f"[INFO] load_composite_table: gene-level composite shape={meta_gene.shape}")
    return meta_gene


# ============================================================
# BrainSpan feature construction from official raw files
# ============================================================

def convert_age_to_weeks(age_str: str) -> float:
    """Convert BrainSpan age text into weeks, following official R logic."""
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
    """
    Smooth one gene's expression in one region and interpolate to 50 time points.

    Mirrors the official idea:
    - LOWESS on log(age)
    - interpolate to fixed 50 points between log(2) and log(2118)
    """
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
    """
    Build the BrainSpan feature matrix directly from official raw ext_data files.

    Output:
    DataFrame indexed by ENSG gene ID, with 800 features (= 16 regions * 50 time points).
    """
    cache_path = ext_data_dir.parent / "cache" / "brainspan_gene_matrix.pkl"
    if cache_path.exists() and not force_rebuild:
        df = load_pickle(cache_path)
        return df.loc[df.index.intersection(pd.Index(sorted(target_proteins)))]

    m_path = ext_data_dir / "brainspan" / "expression_matrix.csv"
    ann_path = ext_data_dir / "brainspan" / "rows_metadata.csv"
    fac_path = ext_data_dir / "brainspan" / "columns_metadata.csv"
    egmap_path = ext_data_dir / "entrez_ids" / "entrezgene2symbol.csv"
    for p, name in [
        (m_path, "BrainSpan expression_matrix.csv"),
        (ann_path, "BrainSpan rows_metadata.csv"),
        (fac_path, "BrainSpan columns_metadata.csv"),
        (egmap_path, "entrezgene2symbol.csv"),
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
    region_names: list[str] = []

    for region in keep_structures:
        mask = (structure == region).to_numpy()
        X = M[:, mask]
        x_age = age_wk[mask]
        region_out = np.zeros((X.shape[0], 50), dtype=np.float32)
        for i in range(X.shape[0]):
            region_out[i, :] = lowess_interpolate(X[i, :], x_age)
        expr_by_region.append(region_out)
        region_names.append(region)

    n_regions = len(expr_by_region)
    if n_regions == 0:
        raise ValueError("No BrainSpan structures passed the >20 samples filter")

    # Stack into gene-level matrices: genes x regions x 50
    gene_mats = np.stack(expr_by_region, axis=1)  # shape: genes x regions x 50

    # Standardize within each gene over the flattened vector, matching the official idea.
    gene_mats_flat = gene_mats.reshape(gene_mats.shape[0], -1).astype(np.float32)
    mean = gene_mats_flat.mean(axis=1, keepdims=True)
    std = gene_mats_flat.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    gene_mats_flat = (gene_mats_flat - mean) / std

    mappings = load_gene_mappings(ext_data_dir)
    entrez_to_ensg = mappings["entrez_to_ensg"]
    symbol_to_ensg = mappings["symbol_to_ensg"]
    genes: list[str] = []
    for e, sym in zip(row_entrez, ann["gene_symbol"].tolist()):
        candidates = set(entrez_to_ensg.get(normalize_value(e).replace(".0", ""), set()))
        if not candidates:
            candidates = set(symbol_to_ensg.get(normalize_value(sym).upper(), set()))
        genes.append(sorted(candidates)[0] if candidates else "")
    valid = [i for i, g in enumerate(genes) if g and g in target_proteins]

    df = pd.DataFrame(
        gene_mats_flat[valid, :],
        index=[genes[i] for i in valid],
        columns=[f"bs_{j+1}" for j in range(gene_mats_flat.shape[1])],
    )
    # Match R na.roughfix-ish behavior for remaining missing values.
    df = df.fillna(df.median(numeric_only=True))
    df = df.groupby(df.index).mean(numeric_only=True)

    save_pickle(df, cache_path)
    return df


# ============================================================
# STRING graph and fold-specific STRING features
# ============================================================

def build_string_graph(ext_data_dir: Path, force_rebuild: bool = False) -> nx.Graph:
    """
    Build a gene-level STRING graph from 9606.protein.links.v10.txt.gz using score > 400.
    """
    return build_gene_string_graph(ext_data_dir, force_rebuild=force_rebuild)



def build_fold_string_feature_matrix(
    G: nx.Graph,
    target_ids: list[str],
    anchor_ids: list[str],
    max_anchors: int = 256, # max to 1000
) -> pd.DataFrame:
    """
    Build fold-specific STRING features using shortest-path distances to training anchors.

    Why this design:
    - The official R pipeline uses a precomputed all-pairs shortest-path matrix.
    - You do not have that .Rdata, only raw official ext_data.
    - In pure Python, this fold-specific anchor-distance matrix keeps the same idea
      while remaining practical on the uploaded official inputs.

    Output:
    DataFrame indexed by target_ids and with one column per anchor protein.
    Values are graph distances; unreachable nodes are assigned a large fallback value.
    """
    anchors = [a for a in anchor_ids if a in G]
    if len(anchors) == 0:
        raise ValueError("No training anchors are present in STRING graph")

    # Keep deterministic subset if anchors are too many.
    anchors = sorted(set(anchors))[:max_anchors]
    target_set = set(target_ids)

    # Large fallback distance for unreachable node pairs.
    unreachable = 9999
    data = {anchor: np.full(len(target_ids), unreachable, dtype=np.int32) for anchor in anchors}
    index_lookup = {pid: i for i, pid in enumerate(target_ids)}

    for anchor in anchors:
        lengths = nx.single_source_shortest_path_length(G, anchor)
        for node, dist in lengths.items():
            j = index_lookup.get(node)
            if j is not None:
                data[anchor][j] = int(dist)

    df = pd.DataFrame(data, index=target_ids)
    return df


# ============================================================
# Model training helpers
# ============================================================

def fit_rf_and_score(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_all: pd.DataFrame,
    n_estimators: int,
    random_state: int,
) -> tuple[object, pd.Series]:
    """Fit a classifier and return P(y=1) scores for all rows."""
    y_unique = np.unique(y_train)
    if y_unique.size < 2:
        constant = int(y_unique[0]) if y_unique.size == 1 else 0
        clf = DummyClassifier(strategy="constant", constant=constant)
        clf.fit(X_train, y_train)
        scores = np.full(X_all.shape[0], float(constant), dtype=float)
        return clf, pd.Series(scores, index=X_all.index)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_all)
    if proba.shape[1] == 1:
        # Defensive: some sklearn classifiers can end up single-class in a fold.
        scores = np.ones(X_all.shape[0], dtype=float) if int(clf.classes_[0]) == 1 else np.zeros(X_all.shape[0], dtype=float)
    else:
        pos_idx = np.flatnonzero(clf.classes_ == 1)
        scores = proba[:, int(pos_idx[0])] if len(pos_idx) else np.zeros(X_all.shape[0], dtype=float)
    return clf, pd.Series(scores, index=X_all.index)


# ============================================================
# CV pipeline
# ============================================================

def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Compute evaluation metrics."""
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


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ks: tuple[int, ...] = (10, 20, 50),
) -> dict[str, float]:
    """Compute ranking metrics on top-K lists (precision/recall/lift/NDCG)."""
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



def run_cv(
    labels_df: pd.DataFrame,
    brainspan_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    G: nx.Graph,
    output_dir: Path,
    n_splits: int = 5,
    random_state: int = 42,
    max_string_anchors: int = 256,
    noise_type: str = "none",
    noise_level: float = 0.0,
    label_flip_rate: float = 0.0,
    label_noise_mode: str = "none",
    label_noise_rate: float = 0.0,
    label_budget_positive_fraction: float = 1.0,
    label_budget_neg_ratio: float = 0.0,
) -> None:
    """
    Run 5-fold CV using the forecASD idea in pure Python.

    Per fold:
    1. Train BrainSpan RF on official BrainSpan-derived features
    2. Train STRING RF on shortest-path-to-anchor features built from official STRING file
    3. Train final RF on [STRING_score, BrainSpan_score] + official composite predictors
    4. Evaluate on the held-out 20%
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_df = labels_df.reset_index(drop=True)

    target_ids = meta_df.index.astype(str).tolist()
    brainspan_df = brainspan_df.loc[brainspan_df.index.intersection(meta_df.index)]

    # We need IDs that exist in all components used for scoring.
    # Use IDs present in composite_table and STRING graph.
    # BrainSpan coverage can be sparse; we handle missing BrainSpan rows with imputation.
    valid_ids = set(meta_df.index) & set(G.nodes)
    labels_df = labels_df[labels_df["id"].isin(valid_ids)].reset_index(drop=True)

    n_pos = int((labels_df["label"] == 1).sum())
    n_neg = int((labels_df["label"] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "After filtering to IDs present in composite_table and STRING graph, only one class remains. "
            f"Kept n_pos={n_pos}, n_neg={n_neg}. Check your SFARI/neg labels mapping and ID overlap."
        )

    labels_df.to_csv(output_dir / "all_labels_used.csv", index=False)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X = labels_df["id"].values
    y = labels_df["label"].values

    all_metrics: list[dict[str, float]] = []
    label_ids = set(labels_df["id"].astype(str))
    full_scores_unlabeled: list[pd.DataFrame] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
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
        pos_train_ids = train_df.loc[train_df["label"] == 1, "id"].tolist()
        y_train_bool = train_df["label"].to_numpy(dtype=int)

        rng = np.random.default_rng(random_state + fold_idx)
        if label_flip_rate > 0.0:
            y_train_bool = _apply_label_noise(y_train_bool, label_flip_rate, rng)

        # ----------------------------------------------------
        # BrainSpan RF (official idea from official raw files)
        # ----------------------------------------------------
        if brainspan_df.empty:
            raise ValueError("BrainSpan feature matrix is empty; check BrainSpan inputs and mapping.")

        bs_median = brainspan_df.median(numeric_only=True)
        X_bs_train = brainspan_df.reindex(train_ids).fillna(bs_median)
        X_bs_all = brainspan_df.reindex(meta_df.index).fillna(bs_median)
        X_bs_train_noisy = pd.DataFrame(
            _apply_feature_noise(X_bs_train.to_numpy(dtype=np.float32), noise_type, noise_level, rng),
            index=X_bs_train.index, columns=X_bs_train.columns,
        ) if noise_type != "none" and noise_level > 0.0 else X_bs_train
        _, brainspan_scores = fit_rf_and_score(
            X_train=X_bs_train_noisy,
            y_train=y_train_bool,
            X_all=X_bs_all,
            n_estimators=1000,
            random_state=5136,
        )

        # Align to all meta rows, fill missing with median score.
        brainspan_scores_all = pd.Series(index=meta_df.index, dtype=float)
        brainspan_scores_all.loc[brainspan_scores.index] = brainspan_scores.values
        brainspan_scores_all = brainspan_scores_all.fillna(brainspan_scores_all.median())

        # ----------------------------------------------------
        # STRING RF (same idea, fold-specific anchor distances)
        # ----------------------------------------------------
        string_feature_df = build_fold_string_feature_matrix(
            G=G,
            target_ids=list(meta_df.index),
            anchor_ids=train_ids,
            max_anchors=max_string_anchors,
        )
        X_str_train = string_feature_df.loc[train_ids]
        X_str_all = string_feature_df
        X_str_train_noisy = pd.DataFrame(
            _apply_feature_noise(X_str_train.to_numpy(dtype=np.float32), noise_type, noise_level, rng),
            index=X_str_train.index, columns=X_str_train.columns,
        ) if noise_type != "none" and noise_level > 0.0 else X_str_train
        _, string_scores = fit_rf_and_score(
            X_train=X_str_train_noisy,
            y_train=y_train_bool,
            X_all=X_str_all,
            n_estimators=500,
            random_state=2176,
        )

        # ----------------------------------------------------
        # Final forecASD-style model
        # ----------------------------------------------------
        work_meta = meta_df.copy()
        work_meta.insert(0, "BrainSpan_score", brainspan_scores_all.loc[work_meta.index].astype(float))
        work_meta.insert(0, "STRING_score", string_scores.loc[work_meta.index].astype(float))

        # Match the official R code: meta[ ids, -(3:9) ] after adding the two scores.
        # Columns 3:9 are ensembl_string through pLI in the official composite table layout.
        cols = work_meta.columns.tolist()
        keep_cols = [c for i, c in enumerate(cols, start=1) if not (3 <= i <= 9)]
        X_final_all = work_meta[keep_cols].apply(pd.to_numeric, errors="coerce")
        X_final_all = X_final_all.fillna(X_final_all.median(numeric_only=True))

        X_final_train = X_final_all.loc[train_ids]
        X_final_train_noisy = pd.DataFrame(
            _apply_feature_noise(X_final_train.to_numpy(dtype=np.float32), noise_type, noise_level, rng),
            index=X_final_train.index, columns=X_final_train.columns,
        ) if noise_type != "none" and noise_level > 0.0 else X_final_train
        _, final_scores = fit_rf_and_score(
            X_train=X_final_train_noisy,
            y_train=y_train_bool,
            X_all=X_final_all,
            n_estimators=500,
            random_state=43775,
        )

        # ----------------------------------------------------
        # Evaluate on held-out 20%
        # ----------------------------------------------------
        test_scores = final_scores.loc[test_df["id"]].to_numpy(dtype=float)
        test_metrics = evaluate_predictions(test_df["label"].to_numpy(dtype=int), test_scores)
        test_metrics["fold"] = fold_idx
        test_metrics["n_test"] = int(len(test_df))
        all_metrics.append(test_metrics)

        # Save fold outputs
        fold_pred_df = test_df.copy()
        fold_pred_df["forecASD"] = test_scores
        fold_pred_df["pred_label"] = (fold_pred_df["forecASD"] >= 0.5).astype(int)
        fold_pred_df.to_csv(fold_dir / "test_predictions.csv", index=False)

        full_scores_df = pd.DataFrame({
            "gene_id": final_scores.index,
            "ensembl_string": final_scores.index,  # legacy alias for older analysis scripts
            "forecASD": final_scores.values,
            "STRING_score": string_scores.loc[final_scores.index].values,
            "BrainSpan_score": brainspan_scores_all.loc[final_scores.index].values,
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
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Pure Python forecASD-style pipeline using official ext_data inputs and new label logic."
    )
    p.add_argument("--project-root", type=str, required=True, help="Directory containing ext_data/ or ext_data.zip")
    p.add_argument("--sfari-new", type=str, default=None, help="Latest SFARI file; defaults to ext_data/SFARI_new.csv")
    p.add_argument("--sfari-gene-ids", type=str, default=None, help="SFARI gene mapping file; defaults to ext_data/sfari_gene_ids_v10.txt")
    p.add_argument(
        "--neg-strategy",
        type=str,
        default="deepnd_reference_fill",
        choices=["deepnd_reference_fill", "sfari_reference_random", "simple_ratio", "sfari_string_aware"],
        help=(
            "Negative label strategy. "
            "'deepnd_reference_fill': use mapped DeepND ASD negatives first, then fill from SFARI-filtered random pool. "
            "'sfari_reference_random': exclude SFARI-risk genes then random sample. "
            "'sfari_string_aware': additionally stratify by STRING network distance."
        ),
    )
    p.add_argument("--target-neg-ratio", type=float, default=3.0, help="Target final negative:positive ratio")
    p.add_argument(
        "--deepnd-negative-ids",
        type=str,
        default=None,
        help=(
            "DeepND ASD negative IDs, one identifier per line. ENSG, ENSP, symbol, and Entrez IDs are mapped to ENSG. "
            "Defaults to ext_data/deepnd_asd_negative_protein_ids_usable_no_pos_overlap.txt"
        ),
    )
    p.add_argument("--tier-a-ratio", type=float, default=0.35,
                   help="Fraction of negatives drawn from STRING neighbours of positives "
                        "(tier A, hard negatives). Only used with --neg-strategy sfari_string_aware.")
    p.add_argument("--tier-a-max-hops", type=int, default=2,
                   help="Max hops from any positive in STRING to define tier A. "
                        "Only used with --neg-strategy sfari_string_aware.")
    p.add_argument("--neg-random-state", type=int, default=42, help="Random seed for negative sampling")
    p.add_argument("--sfari-report-min", type=int, default=3, help="Exclude SFARI genes with number-of-reports >= this threshold")
    p.add_argument("--sfari-eagle-min", type=float, default=1.0, help="Exclude SFARI genes with eagle >= this threshold")
    p.add_argument("--output-dir", type=str, default="forecasd_outputs", help="Output directory name under project root")
    p.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for CV splitting")
    p.add_argument("--max-string-anchors", type=int, default=256, help="Max training anchors used to build STRING shortest-path features per fold")
    p.add_argument("--force-rebuild-brainspan", action="store_true", help="Rebuild cached BrainSpan matrix")
    p.add_argument("--force-rebuild-string", action="store_true", help="Rebuild cached STRING graph")
    p.add_argument("--noise-type", type=str, default="none", choices=["none", "gaussian", "dropout"],
                   help="Feature noise type applied to training data only (default: none)")
    p.add_argument("--noise-level", type=float, default=0.0,
                   help="Noise level: std multiplier for gaussian, drop rate for dropout (default: 0.0)")
    p.add_argument("--label-flip-rate", type=float, default=0.0,
                   help="Fraction of negative training labels flipped to positive (default: 0.0)")
    add_label_noise_args(p)
    add_label_budget_args(p)
    return p.parse_args()


def main():

    args = parse_args()

    project_root = Path(args.project_root)

    ext_data_dir = project_root / "ext_data"

    ensure_exists(ext_data_dir, "ext_data directory")

    sfari_new = Path(args.sfari_new).resolve() if args.sfari_new else ext_data_dir / "SFARI_new.csv"
    sfari_gene_ids = Path(args.sfari_gene_ids).resolve() if args.sfari_gene_ids else ext_data_dir / "sfari_gene_ids_v10.txt"
    deepnd_negative_ids_path = (
        Path(args.deepnd_negative_ids).resolve()
        if args.deepnd_negative_ids
        else ext_data_dir / "deepnd_asd_negative_protein_ids_usable_no_pos_overlap.txt"
    )
    ensure_exists(sfari_new, "SFARI_new.csv")
    ensure_exists(sfari_gene_ids, "sfari_gene_ids_v10.txt")

    print("[INFO] Loading official composite table...")
    meta_df = load_composite_table(ext_data_dir)

    print("[INFO] Building STRING graph from official raw file...")
    G = build_string_graph(ext_data_dir, force_rebuild=args.force_rebuild_string)
    print(f"[INFO] STRING graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

    print("[INFO] Building ID mappings...")
    mapping = build_mapping_dicts(sfari_gene_ids)

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Building BrainSpan matrix from official raw files...")
    brainspan_df = build_brainspan_matrix(
        ext_data_dir=ext_data_dir,
        target_proteins=set(meta_df.index.astype(str)),
        force_rebuild=args.force_rebuild_brainspan,
    )
    print(f"[INFO] BrainSpan matrix shape: {brainspan_df.shape}")

    print("[INFO] Generating labels with new logic...")
    pos_ids = extract_positive_ids_from_sfari(sfari_new, mapping)
    valid_ids = set(meta_df.index.astype(str)) & set(G.nodes)
    target_neg_ratio = float(args.target_neg_ratio)

    sfari_reference_excluded_ids, sfari_reference_df = extract_sfari_reference_exclusion_ids(
        sfari_new_path=sfari_new,
        mapping=mapping,
        report_min=int(args.sfari_report_min),
        eagle_min=float(args.sfari_eagle_min),
        exclude_syndromic=True,
    )

    if args.neg_strategy == "deepnd_reference_fill":
        print(
            "[INFO] DeepND negative selection: "
            f"use published mapped negatives first, fill to "
            f"{target_neg_ratio:.2f}:1 with SFARI-reference random negatives"
        )
        deepnd_neg_ids = load_deepnd_negative_ids(deepnd_negative_ids_path)
        labels_df, selected_neg_ids, neg_report = select_negatives_deepnd_then_reference_fill(
            pos_ids=pos_ids,
            deepnd_neg_ids=deepnd_neg_ids,
            reference_exclude_ids=sfari_reference_excluded_ids,
            valid_ids=valid_ids,
            target_neg_ratio=target_neg_ratio,
            random_state=args.neg_random_state,
        )
        sfari_reference_df.to_csv(output_dir / "sfari_reference_excluded_genes.csv", index=False)
        (output_dir / "sfari_reference_excluded_ids.txt").write_text(
            "\n".join(sorted(sfari_reference_excluded_ids & valid_ids)), encoding="utf-8",
        )
        selected_deepnd = set(deepnd_neg_ids) & selected_neg_ids
        selected_fill = selected_neg_ids - selected_deepnd
        (output_dir / "neg_deepnd_selected_ids.txt").write_text(
            "\n".join(sorted(selected_deepnd)), encoding="utf-8",
        )
        (output_dir / "neg_fill_random_ids.txt").write_text(
            "\n".join(sorted(selected_fill)), encoding="utf-8",
        )
        (output_dir / "neg_candidate_pool_ids.txt").write_text(
            "\n".join(sorted(valid_ids - set(pos_ids) - (sfari_reference_excluded_ids & valid_ids) - selected_deepnd)),
            encoding="utf-8",
        )
        (output_dir / "neg_selected_random_ids.txt").write_text(
            "\n".join(sorted(selected_neg_ids)), encoding="utf-8",
        )
        with open(output_dir / "neg_selection_report.json", "w", encoding="utf-8") as f:
            json.dump(neg_report, f, ensure_ascii=False, indent=2)
    elif args.neg_strategy == "sfari_string_aware":
        print(
            "[INFO] STRING-aware stratified negative selection: "
            f"SFARI exclusion + tier_a_ratio={args.tier_a_ratio:.2f} "
            f"(STRING {args.tier_a_max_hops}-hop neighbours of positives), "
            f"target ratio={target_neg_ratio:.2f}:1"
        )
        labels_df, selected_neg_ids, neg_report = select_negatives_random_from_reference_filtered(
            pos_ids=pos_ids,
            reference_exclude_ids=sfari_reference_excluded_ids,
            valid_ids=valid_ids,
            target_neg_ratio=target_neg_ratio,
            random_state=args.neg_random_state,
            G=G,
            tier_a_ratio=args.tier_a_ratio,
            tier_a_max_hops=args.tier_a_max_hops,
        )
        sfari_reference_df.to_csv(output_dir / "sfari_reference_excluded_genes.csv", index=False)
        (output_dir / "sfari_reference_excluded_ids.txt").write_text(
            "\n".join(sorted(sfari_reference_excluded_ids & valid_ids)), encoding="utf-8",
        )
        (output_dir / "neg_candidate_pool_ids.txt").write_text(
            "\n".join(sorted(valid_ids - set(pos_ids) - (sfari_reference_excluded_ids & valid_ids))),
            encoding="utf-8",
        )
        (output_dir / "neg_selected_random_ids.txt").write_text(
            "\n".join(sorted(selected_neg_ids)), encoding="utf-8",
        )
        with open(output_dir / "neg_selection_report.json", "w", encoding="utf-8") as f:
            json.dump(neg_report, f, ensure_ascii=False, indent=2)
    elif args.neg_strategy == "sfari_reference_random":
        print(
            "[INFO] SFARI-reference random negative selection: "
            f"exclude SFARI-risk genes by score/syndromic/reports/eagle, "
            f"target ratio={target_neg_ratio:.2f}:1"
        )
        labels_df, selected_neg_ids, neg_report = select_negatives_random_from_reference_filtered(
            pos_ids=pos_ids,
            reference_exclude_ids=sfari_reference_excluded_ids,
            valid_ids=valid_ids,
            target_neg_ratio=target_neg_ratio,
            random_state=args.neg_random_state,
        )
        sfari_reference_df.to_csv(output_dir / "sfari_reference_excluded_genes.csv", index=False)
        (output_dir / "sfari_reference_excluded_ids.txt").write_text(
            "\n".join(sorted(sfari_reference_excluded_ids & valid_ids)),
            encoding="utf-8",
        )
        (output_dir / "neg_candidate_pool_ids.txt").write_text(
            "\n".join(sorted(valid_ids - set(pos_ids) - (sfari_reference_excluded_ids & valid_ids))),
            encoding="utf-8",
        )
        (output_dir / "neg_selected_random_ids.txt").write_text(
            "\n".join(sorted(selected_neg_ids)),
            encoding="utf-8",
        )
        with open(output_dir / "neg_selection_report.json", "w", encoding="utf-8") as f:
            json.dump(neg_report, f, ensure_ascii=False, indent=2)
    else:
        print(
            "[INFO] Simple-ratio negative selection from random eligible pool: "
            f"ratio={target_neg_ratio:.2f}:1"
        )
        labels_df, _, neg_report = select_negatives_random_from_reference_filtered(
            pos_ids=pos_ids,
            reference_exclude_ids=sfari_reference_excluded_ids,
            valid_ids=valid_ids,
            target_neg_ratio=target_neg_ratio,
            random_state=args.neg_random_state,
        )
        sfari_reference_df.to_csv(output_dir / "sfari_reference_excluded_genes.csv", index=False)
        with open(output_dir / "neg_selection_report.json", "w", encoding="utf-8") as f:
            json.dump(neg_report, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Positive IDs kept: {int(labels_df['label'].sum())}")
    print(f"[INFO] Negative IDs kept: {int((labels_df['label'] == 0).sum())}")

    save_label_outputs(labels_df, output_dir)
    print("[INFO] Running pure Python forecASD-style CV...")
    run_cv(
        labels_df=labels_df,
        brainspan_df=brainspan_df,
        meta_df=meta_df,
        G=G,
        output_dir=output_dir,
        n_splits=args.n_splits,
        random_state=args.random_state,
        label_noise_mode=args.label_noise_mode,
        label_noise_rate=args.label_noise_rate,
        label_budget_positive_fraction=args.label_budget_positive_fraction,
        label_budget_neg_ratio=args.label_budget_neg_ratio,
        max_string_anchors=args.max_string_anchors,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
        label_flip_rate=args.label_flip_rate,
    )

    print(f"[DONE] Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

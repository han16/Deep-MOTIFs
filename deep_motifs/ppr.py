from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# v16: Personalized PageRank score propagation on STRING graph
# ============================================================

def propagate_scores_ppr(
    scores: pd.Series,
    G,
    alpha: float = 0.5,
    n_iter: int = 30,
    min_edge_weight: float = 0.5,
) -> pd.Series:
    """
    Personalized PageRank (Random Walk with Restart) on the STRING network.

    r_{t+1} = alpha * p0 + (1 - alpha) * A_norm @ r_t

    alpha         : restart probability. Higher 鈫?stays closer to original scores.
                    alpha=1.0 means no propagation (identity).
    min_edge_weight: filter out STRING edges below this normalised weight before
                    propagation. Default 0.5 corresponds to STRING score > 700
                    when the graph was built with base threshold=400
                    (weight = (score-400)/600, so 0.5 鈫?score > 700).

    Genes not present in the STRING graph keep their original score unchanged.
    Output is min-max normalised to [1e-6, 1-1e-6].
    """
    import scipy.sparse as sp

    # Restrict to genes present in both scores index and graph
    nodes = [n for n in scores.index if n in G]
    if not nodes:
        return scores.copy()

    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Build sparse weighted adjacency (symmetric, filtered by weight)
    rows, cols, data = [], [], []
    for u, v, d in G.edges(nbunch=nodes, data=True):
        if u not in node_idx or v not in node_idx:
            continue
        w = d.get("weight", 1.0)
        if w < min_edge_weight:
            continue
        i, j = node_idx[u], node_idx[v]
        rows += [i, j]
        cols += [j, i]
        data += [w, w]

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)

    # Row-normalise 鈫?stochastic transition matrix
    row_sums = np.asarray(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    A_norm = sp.diags(1.0 / row_sums) @ A

    # Personalisation vector: original scores (non-negative, normalised to sum=1)
    p0 = scores.reindex(nodes).fillna(0.0).to_numpy(dtype=np.float64)
    p0 = np.clip(p0, 0.0, None)
    p0_sum = p0.sum()
    if p0_sum > 0:
        p0 = p0 / p0_sum
    else:
        p0 = np.ones(n, dtype=np.float64) / n

    # Power iteration
    r = p0.copy()
    for _ in range(n_iter):
        r_new = alpha * p0 + (1.0 - alpha) * A_norm.dot(r)
        if np.max(np.abs(r_new - r)) < 1e-8:
            r = r_new
            break
        r = r_new

    # Min-max normalise to [1e-6, 1-1e-6]
    r_min, r_max = r.min(), r.max()
    if r_max > r_min:
        r = (r - r_min) / (r_max - r_min)
    r = np.clip(r, 1e-6, 1.0 - 1e-6)

    # Write back; genes absent from graph keep original score
    result = scores.copy().astype(float)
    for node, i in node_idx.items():
        result[node] = float(r[i])
    return result


def compute_ppr_from_seeds(
    seed_ids: list,
    all_ids: list,
    G,
    alpha: float = 0.5,
    n_iter: int = 30,
    min_edge_weight: float = 0.5,
    seed_weights: dict | None = None,
) -> pd.Series:
    """
    v17: Seeded Personalized PageRank.

    p鈧€ = uniform distribution over seed_ids (known positive genes).
    All other genes start at 0.

    v20: seed_weights (optional dict {gene_id: weight}) allows confidence-weighted
    seeds. If provided, each seed's initial probability is proportional to its
    weight (e.g. XGBoost OOF score). High-confidence positives contribute more
    to the propagation signal, reducing noise from uncertain training positives.

    r_{t+1} = alpha * p0 + (1 - alpha) * A_norm @ r_t

    Returns a pd.Series indexed by all_ids representing each gene's
    network proximity to the seed (known ASD) genes.
    Genes absent from the STRING graph receive score 0.0.
    Output is min-max normalised to [1e-6, 1-1e-6].
    """
    import scipy.sparse as sp

    seed_set = set(seed_ids)
    nodes = [n for n in all_ids if n in G]
    if not nodes or not seed_set:
        return pd.Series(0.0, index=all_ids)

    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Build sparse weighted adjacency (symmetric, filtered by weight)
    rows, cols, data = [], [], []
    for u, v, d in G.edges(nbunch=nodes, data=True):
        if u not in node_idx or v not in node_idx:
            continue
        w = d.get("weight", 1.0)
        if w < min_edge_weight:
            continue
        i, j = node_idx[u], node_idx[v]
        rows += [i, j]
        cols += [j, i]
        data += [w, w]

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)

    # Row-normalise 鈫?stochastic transition matrix
    row_sums = np.asarray(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    A_norm = sp.diags(1.0 / row_sums) @ A

    # Personalisation vector: uniform (v17) or confidence-weighted (v20) seeds
    seeds_in_graph = [s for s in seed_ids if s in node_idx]
    p0 = np.zeros(n, dtype=np.float64)
    if seeds_in_graph:
        if seed_weights is not None:
            for s in seeds_in_graph:
                p0[node_idx[s]] = seed_weights.get(s, 1.0)
        else:
            for s in seeds_in_graph:
                p0[node_idx[s]] = 1.0
        p0 /= p0.sum()
    else:
        p0 = np.ones(n, dtype=np.float64) / n

    # Power iteration
    r = p0.copy()
    for _ in range(n_iter):
        r_new = alpha * p0 + (1.0 - alpha) * A_norm.dot(r)
        if np.max(np.abs(r_new - r)) < 1e-8:
            r = r_new
            break
        r = r_new

    # Min-max normalise to [1e-6, 1-1e-6]
    r_min, r_max = r.min(), r.max()
    if r_max > r_min:
        r = (r - r_min) / (r_max - r_min)
    r = np.clip(r, 1e-6, 1.0 - 1e-6)

    # Build full result indexed by all_ids; genes not in graph get 0.0
    result = pd.Series(0.0, index=all_ids, dtype=float)
    for node, i in node_idx.items():
        result[node] = float(r[i])
    return result
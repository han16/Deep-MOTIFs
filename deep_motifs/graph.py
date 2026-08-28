from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from .gene_id_utils import load_gene_mappings


def build_weighted_string_graph(
    ext_data_dir: Path,
    score_threshold: int = 400,
    cache_path: Path | None = None,
) -> "nx.Graph":
    """
    Re-read the raw STRING file, storing normalised weights on the edges.
    weight = (score - threshold) / (1000 - threshold), range (0, 1].

    build_string_graph in xgb.py discards the weights; here we read them
    separately so that the weight information is preserved for the weighted GCN.
    Uses its own cache file, so xgb.py's cache is not affected.
    """
    import networkx as nx
    import pickle

    if cache_path is not None and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    mappings = load_gene_mappings(ext_data_dir)
    ensp_to_ensg = mappings["ensp_to_ensg"]
    string_path = ext_data_dir / "9606.protein.links.v10.txt.gz"
    if not string_path.exists():
        raise FileNotFoundError(f"STRING file not found: {string_path}")

    G = nx.Graph()
    with gzip.open(string_path, "rt") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split()
            a, b, score = parts[0], parts[1], int(parts[2])
            if score <= score_threshold:
                continue
            a = a.replace("9606.", "")
            b = b.replace("9606.", "")
            ga = ensp_to_ensg.get(a)
            gb = ensp_to_ensg.get(b)
            if not ga or not gb or ga == gb:
                continue
            w = float(score - score_threshold) / float(1000 - score_threshold)
            if G.has_edge(ga, gb):
                if w > float(G[ga][gb].get("weight", 0.0)):
                    G[ga][gb]["weight"] = w
            else:
                G.add_edge(ga, gb, weight=w)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(G, f)
        print(f"[INFO] Weighted STRING graph cached to: {cache_path}")

    return G


def _gcn_aggregate_string(
    x_str: np.ndarray,
    G,
    ids_all: list[str],
    allowed_ids: set[str],
    n_layers: int = 2,
    self_weight: float = 0.5,
) -> np.ndarray:
    """
    n_layers of weighted GCN pre-aggregation:
        x_agg[i] = self_weight x x[i]
                 + (1 - self_weight) x sum(w_ij x x[j]) / sum(w_ij)

    Edge weight w_ij comes from the normalised STRING score.
    If the graph has no weight attribute (fallback to an unweighted graph),
    aggregation is done with equal weights.
    Neighbour aggregation is restricted to allowed_ids (the train universe)
    to avoid leaking test information.
    """
    lookup  = {pid: i for i, pid in enumerate(ids_all)}
    allowed = set(allowed_ids)
    x       = x_str.copy().astype(np.float32)

    for _layer in range(n_layers):
        x_new = x.copy()
        for pid, i in lookup.items():
            if pid not in G or pid not in allowed:
                continue
            # collect the weighted neighbours
            nbs: list[tuple[int, float]] = []
            for n in G.neighbors(pid):
                if n not in lookup or n not in allowed:
                    continue
                # prefer the edge weight; default to 1.0 when unweighted
                w = float(G[pid][n].get("weight", 1.0))
                nbs.append((lookup[n], w))
            if not nbs:
                continue
            total_w = sum(w for _, w in nbs)
            if total_w <= 0:
                continue
            nb_agg   = sum(w * x[j] for j, w in nbs) / total_w
            x_new[i] = self_weight * x[i] + (1.0 - self_weight) * nb_agg
        x = x_new

    return x.astype(np.float32)

def build_neighbor_matrix(
    G,
    ids: list[str],
    allowed_ids: set[str],
    top_k: int = 3,
) -> np.ndarray:
    lookup  = {pid: i for i, pid in enumerate(ids)}
    allowed = set(allowed_ids)
    matrix  = np.full((len(ids), top_k), -1, dtype=np.int64)
    for pid, i in lookup.items():
        if pid not in G or pid not in allowed:
            continue
        nbs = [
            (n, G.degree(n))
            for n in G.neighbors(pid)
            if n in lookup and n in allowed
        ]
        if not nbs:
            continue
        nbs.sort(key=lambda x: -x[1])
        for k_idx, (n, _) in enumerate(nbs[:top_k]):
            matrix[i, k_idx] = lookup[n]
    return matrix

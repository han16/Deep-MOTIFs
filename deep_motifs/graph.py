from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np


# ============================================================
# 鏀瑰姩涓€锛氬甫鏉冮噸鐨?STRING 鍥?+ 2灞傚姞鏉?GCN 棰勮仛鍚?# ============================================================

def build_weighted_string_graph(
    data_dir: Path,
    score_threshold: int = 400,
    cache_path: Path | None = None,
) -> "nx.Graph":
    """
    閲嶆柊璇诲彇 STRING 鍘熷鏂囦欢锛屽湪杈逛笂瀛樺偍褰掍竴鍖栨潈閲嶃€?    鏉冮噸 = (score - threshold) / (1000 - threshold)锛岃寖鍥?(0, 1]銆?
    xgb.py 鐨?build_string_graph 涓㈠純浜嗘潈閲嶏紝杩欓噷鐙珛璇诲彇浠ヤ繚鐣欐潈閲嶄俊鎭紝
    渚涘姞鏉?GCN 浣跨敤銆備娇鐢ㄧ嫭绔嬬殑缂撳瓨鏂囦欢锛屼笉褰卞搷 xgb.py 鐨勭紦瀛樸€?    """
    import networkx as nx

    if cache_path is not None and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    string_path = data_dir / "9606.protein.links.v10.txt.gz"
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
            w = float(score - score_threshold) / float(1000 - score_threshold)
            G.add_edge(a, b, weight=w)

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
    n_layers 灞傚姞鏉?GCN 棰勮仛鍚堬細
        x_agg[i] = self_weight 脳 x[i]
                 + (1 - self_weight) 脳 危(w_ij 脳 x[j]) / 危(w_ij)

    杈规潈閲?w_ij 鏉ヨ嚜 STRING score 褰掍竴鍖栧€笺€?    鑻ュ浘鏃犳潈閲嶅睘鎬э紙fallback 鍒版棤鏉冮噸鍥撅級锛屽垯绛夋潈閲嶈仛鍚堛€?    鍙湪 allowed_ids锛坱rain universe锛夊唴鍋氶偦灞呰仛鍚堬紝閬垮厤 test 淇℃伅娉勬紡銆?    """
    lookup  = {pid: i for i, pid in enumerate(ids_all)}
    allowed = set(allowed_ids)
    x       = x_str.copy().astype(np.float32)

    for _layer in range(n_layers):
        x_new = x.copy()
        for pid, i in lookup.items():
            if pid not in G or pid not in allowed:
                continue
            # 鏀堕泦甯︽潈閲嶇殑閭诲眳
            nbs: list[tuple[int, float]] = []
            for n in G.neighbors(pid):
                if n not in lookup or n not in allowed:
                    continue
                # 浼樺厛浣跨敤杈规潈閲嶏紝鏃犳潈閲嶆椂榛樿 1.0
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
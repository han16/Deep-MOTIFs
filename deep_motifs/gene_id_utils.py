from __future__ import annotations

import gzip
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def normalize_value(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    return s


def load_gene_mappings(ext_data_dir: Path) -> dict[str, dict]:
    """Load ENSP/ENSG/gene-symbol mappings from jack and SFARI mapping files."""
    jack_path = ext_data_dir / "jack_fu_gene_info(in).csv"
    if not jack_path.exists():
        raise FileNotFoundError(f"jack_fu_gene_info(in).csv not found: {jack_path}")

    jack = pd.read_csv(jack_path, dtype=str).fillna("")
    required = {"ensembl_gene_id", "ensembl_peptide_id"}
    missing = required - set(jack.columns)
    if missing:
        raise ValueError(f"{jack_path.name} missing columns: {sorted(missing)}")

    ensp_to_ensg: dict[str, str] = {}
    ensg_to_ensps: dict[str, set[str]] = {}
    ensg_to_symbol: dict[str, str] = {}
    symbol_to_ensg: dict[str, set[str]] = {}
    entrez_to_ensg: dict[str, set[str]] = {}

    for _, row in jack.iterrows():
        ensg = normalize_value(row.get("ensembl_gene_id", ""))
        ensp = normalize_value(row.get("ensembl_peptide_id", ""))
        sym = normalize_value(row.get("hgnc_symbol", ""))
        entrez = normalize_value(row.get("entrezgene_id", row.get("entrez_id", "")))
        if ensg and ensp:
            ensp_to_ensg[ensp] = ensg
            ensg_to_ensps.setdefault(ensg, set()).add(ensp)
        if ensg and sym:
            ensg_to_symbol.setdefault(ensg, sym)
            symbol_to_ensg.setdefault(sym.upper(), set()).add(ensg)
        if ensg and entrez:
            entrez_to_ensg.setdefault(entrez.replace(".0", ""), set()).add(ensg)

    sfari_path = ext_data_dir / "sfari_gene_ids_v10.txt"
    if sfari_path.exists():
        sf = pd.read_csv(sfari_path, sep="\t", dtype=str).fillna("")
        for _, row in sf.iterrows():
            ensg = normalize_value(row.get("Gene stable ID", ""))
            ensp = normalize_value(row.get("Protein stable ID", ""))
            sym = normalize_value(row.get("Gene name", ""))
            entrez = normalize_value(row.get("NCBI gene ID", ""))
            if ensg and ensp:
                ensp_to_ensg.setdefault(ensp, ensg)
                ensg_to_ensps.setdefault(ensg, set()).add(ensp)
            if ensg and sym:
                ensg_to_symbol.setdefault(ensg, sym)
                symbol_to_ensg.setdefault(sym.upper(), set()).add(ensg)
            if ensg and entrez:
                entrez_to_ensg.setdefault(entrez.replace(".0", ""), set()).add(ensg)

    return {
        "ensp_to_ensg": ensp_to_ensg,
        "ensg_to_ensps": ensg_to_ensps,
        "ensg_to_symbol": ensg_to_symbol,
        "symbol_to_ensg": symbol_to_ensg,
        "entrez_to_ensg": entrez_to_ensg,
    }


def map_identifiers_to_genes(
    identifiers: list[object] | pd.Series,
    mappings: dict[str, dict],
) -> set[str]:
    """Map ENSG, ENSP, symbols, or Entrez IDs to ENSG gene IDs."""
    out: set[str] = set()
    for raw in identifiers:
        key = normalize_value(raw)
        if not key:
            continue
        key_u = key.upper()
        if key.startswith("ENSG"):
            out.add(key)
        if key in mappings["ensp_to_ensg"]:
            out.add(mappings["ensp_to_ensg"][key])
        if key_u in mappings["symbol_to_ensg"]:
            out.update(mappings["symbol_to_ensg"][key_u])
        if key.replace(".0", "") in mappings["entrez_to_ensg"]:
            out.update(mappings["entrez_to_ensg"][key.replace(".0", "")])
    return out


def aggregate_table_to_gene_level(
    df: pd.DataFrame,
    ext_data_dir: Path,
    id_column: str | None = None,
) -> pd.DataFrame:
    """Aggregate an ENSP-indexed table to ENSG-indexed gene-level rows."""
    mappings = load_gene_mappings(ext_data_dir)
    ensp_to_ensg = mappings["ensp_to_ensg"]
    ensg_to_symbol = mappings["ensg_to_symbol"]

    work = df.copy()
    if id_column and id_column in work.columns:
        ids = work[id_column].astype(str)
    else:
        ids = work.index.astype(str)

    gene_ids = []
    for raw in ids:
        s = normalize_value(raw)
        if s.startswith("ENSG"):
            gene_ids.append(s)
        else:
            gene_ids.append(ensp_to_ensg.get(s, ""))

    work["_gene_id"] = gene_ids
    work = work[work["_gene_id"] != ""].copy()
    if work.empty:
        raise ValueError("No rows could be mapped to ENSG gene IDs.")

    # Coerce columns independently. Numeric columns are averaged; binary-like
    # numeric columns are maxed; strings keep their first non-empty value.
    out_parts: list[pd.DataFrame] = []
    value_cols = [c for c in work.columns if c != "_gene_id"]
    numeric_cols = []
    string_cols = []
    for col in value_cols:
        coerced = pd.to_numeric(work[col], errors="coerce")
        if coerced.notna().any():
            work[col] = coerced
            numeric_cols.append(col)
        else:
            string_cols.append(col)

    if numeric_cols:
        agg: dict[str, str] = {}
        for col in numeric_cols:
            vals = pd.to_numeric(work[col], errors="coerce").dropna().unique()
            finite_vals = vals[np.isfinite(vals)] if len(vals) else vals
            if len(finite_vals) and set(np.unique(finite_vals)).issubset({0, 1, 0.0, 1.0}):
                agg[col] = "max"
            else:
                agg[col] = "mean"
        out_parts.append(work.groupby("_gene_id")[numeric_cols].agg(agg))

    if string_cols:
        str_df = (
            work.groupby("_gene_id")[string_cols]
            .agg(lambda s: next((normalize_value(x) for x in s if normalize_value(x)), ""))
        )
        out_parts.append(str_df)

    out = pd.concat(out_parts, axis=1) if out_parts else pd.DataFrame(index=sorted(set(gene_ids)))
    out.index = out.index.astype(str)
    out.index.name = None
    out["gene_id"] = out.index
    out["ensembl_string"] = out.index  # legacy column name expected by older feature code
    out["symbol"] = [ensg_to_symbol.get(g, normalize_value(out.loc[g, "symbol"]) if "symbol" in out.columns else "") for g in out.index]

    # Keep a stable front-column layout similar to the original composite table.
    front = [c for c in ["gene_id", "ensembl_string", "entrez", "symbol"] if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].sort_index()


def build_gene_string_graph(ext_data_dir: Path, force_rebuild: bool = False) -> nx.Graph:
    """Collapse STRING ENSP graph to an ENSG gene graph using max edge score."""
    cache_path = ext_data_dir.parent / "cache" / "string_gene_graph.pkl"
    if cache_path.exists() and not force_rebuild:
        with open(cache_path, "rb") as f:
            import pickle
            return pickle.load(f)

    mappings = load_gene_mappings(ext_data_dir)
    ensp_to_ensg = mappings["ensp_to_ensg"]
    string_path = ext_data_dir / "9606.protein.links.v10.txt.gz"
    if not string_path.exists():
        raise FileNotFoundError(f"9606.protein.links.v10.txt.gz not found: {string_path}")

    G = nx.Graph()
    with gzip.open(string_path, "rt") as f:
        _ = next(f)
        for line in f:
            a, b, score_s = line.strip().split()
            score = int(score_s)
            if score <= 400:
                continue
            a = a.replace("9606.", "")
            b = b.replace("9606.", "")
            ga = ensp_to_ensg.get(a)
            gb = ensp_to_ensg.get(b)
            if not ga or not gb or ga == gb:
                continue
            w = float(score) / 1000.0
            if G.has_edge(ga, gb):
                if w > float(G[ga][gb].get("weight", 0.0)):
                    G[ga][gb]["weight"] = w
                    G[ga][gb]["string_score"] = score
            else:
                G.add_edge(ga, gb, weight=w, string_score=score)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        import pickle
        pickle.dump(G, f)
    return G


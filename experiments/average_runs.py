"""
Average cv_metrics_summary.csv and full_scores_summary.csv across run_1..run_5.
Output goes to mean_run/<algo_outputs>/.
"""

from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Project root that contains run_1..run_5 (and where mean_run/ is written).
# Overridable with --project-root; defaults to the current working directory.
ROOT = Path.cwd()
RUNS = [ROOT / f"run_{i}" for i in range(1, 6)]
OUT_ROOT = ROOT / "mean_run"

ALGO_DIRS = [
    "cnn_outputs", "deepgbm_outputs", "forecasd_outputs", "ftt_outputs",
    "gcn_outputs", "deep_motifs_outputs",
    "rf_outputs", "sai_outputs", "sv_outputs", "tab_outputs", "xgb_outputs",
]

# For deep_motifs_outputs (Deep-MOTIFs), also average the global-threshold summary
EXTRA_FILES = {
    "deep_motifs_outputs": ["cv_metrics_summary_global_threshold.csv"],
}


def average_cv_metrics(paths: list[Path]) -> pd.DataFrame:
    """
    cv_metrics_summary.csv has columns: metric, mean, std
    Compute:
      mean_of_means  = average of the 5 run 'mean' values
      mean_of_stds   = average of the 5 run 'std'  values (avg within-fold variation)
      std_across_runs = std of the 5 run 'mean' values (between-seed variation)
    """
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df = df.set_index("metric")
        dfs.append(df)

    metrics = list(dict.fromkeys(m for df in dfs for m in df.index.tolist()))
    rows = []
    for m in metrics:
        means = np.array([df.loc[m, "mean"] if m in df.index else np.nan for df in dfs], dtype=float)
        stds  = np.array([df.loc[m, "std"]  if m in df.index else np.nan for df in dfs], dtype=float)
        valid_means = np.sum(~np.isnan(means))
        valid_stds = np.sum(~np.isnan(stds))
        rows.append({
            "metric":          m,
            "mean":            float(np.nanmean(means)) if valid_means else np.nan,
            "std":             float(np.nanmean(stds)) if valid_stds else np.nan,
            "std_across_runs": float(np.nanstd(means, ddof=1)) if valid_means > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def average_full_scores(paths: list[Path]) -> pd.DataFrame:
    """
    full_scores_summary.csv has columns: ensembl_string, forecASD.
    Average scores across runs; keep all genes present in any run.
    """
    dfs = []
    for p in paths:
        df = pd.read_csv(p).set_index("ensembl_string")
        dfs.append(df["forecASD"].rename(p.parent.parent.name))

    combined = pd.concat(dfs, axis=1)
    result = pd.DataFrame({
        "ensembl_string": combined.index,
        "score_mean": combined.mean(axis=1, skipna=True).to_numpy(),
        "score_std": combined.std(axis=1, skipna=True, ddof=1).fillna(0.0).to_numpy(),
        "n_runs": combined.notna().sum(axis=1).to_numpy(),
    })
    return result


def process_algo(algo: str):
    # Collect existing run paths
    cv_paths    = [r / algo / "cv_metrics_summary.csv"  for r in RUNS]
    score_paths = [r / algo / "full_scores_summary.csv" for r in RUNS]

    cv_ok    = [p for p in cv_paths    if p.exists()]
    score_ok = [p for p in score_paths if p.exists()]

    if not cv_ok:
        print(f"  [SKIP] {algo}: no cv_metrics_summary.csv found in any run")
        return

    out_dir = OUT_ROOT / algo
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- cv_metrics_summary ---
    cv_avg = average_cv_metrics(cv_ok)
    cv_out = out_dir / "cv_metrics_summary.csv"
    cv_avg.to_csv(cv_out, index=False)
    print(f"  [OK]   {algo}/cv_metrics_summary.csv  (from {len(cv_ok)} runs)")

    # --- full_scores_summary ---
    if score_ok:
        score_avg = average_full_scores(score_ok)
        score_out = out_dir / "full_scores_summary.csv"
        score_avg.to_csv(score_out, index=False)
        print(f"  [OK]   {algo}/full_scores_summary.csv  (from {len(score_ok)} runs)")
    else:
        print(f"  [SKIP] {algo}/full_scores_summary.csv: not found in any run")

    # --- extra files (e.g. deep_motifs_outputs / Deep-MOTIFs global threshold) ---
    for extra_file in EXTRA_FILES.get(algo, []):
        extra_paths = [r / algo / extra_file for r in RUNS]
        extra_ok    = [p for p in extra_paths if p.exists()]
        if not extra_ok:
            print(f"  [SKIP] {algo}/{extra_file}: not found in any run")
            continue
        extra_avg = average_cv_metrics(extra_ok)
        extra_out = out_dir / extra_file
        extra_avg.to_csv(extra_out, index=False)
        print(f"  [OK]   {algo}/{extra_file}  (from {len(extra_ok)} runs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Directory containing run_1..run_5 (default: current directory)",
    )
    args = parser.parse_args()

    ROOT = Path(args.project_root).resolve()
    RUNS = [ROOT / f"run_{i}" for i in range(1, 6)]
    OUT_ROOT = ROOT / "mean_run"

    print(f"Averaging across: {[r.name for r in RUNS if r.exists()]}")
    print(f"Output root: {OUT_ROOT}\n")
    for algo in ALGO_DIRS:
        process_algo(algo)
    print("\nDone.")

"""
Add MCC-vs-rank-threshold curve metrics to existing experiment outputs.

For each model output directory containing fold_*/test_predictions.csv, this script:
  - computes Matthews' Correlation Coefficient (MCC) after calling the top X%
    ranked genes positive, for X = 1..100;
  - writes mcc_rank_threshold_curve.csv;
  - appends per-fold mcc_auc/mcc_mean/mcc_max/mcc_at_* metrics to cv_fold_metrics.csv;
  - appends summary rows to cv_metrics_summary.csv.

It also updates mean_run/<algo>/ by averaging the new rows from run_1..run_5.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PCTS = np.arange(1, 101, dtype=float) / 100.0
AT_PCTS = {
    "mcc_at_1pct": 0.01,
    "mcc_at_5pct": 0.05,
    "mcc_at_10pct": 0.10,
    "mcc_at_20pct": 0.20,
}


def matthews_corrcoef_from_counts(tp: int, fp: int, tn: int, fn: int) -> float:
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0:
        return 0.0
    return float((tp * tn - fp * fn) / math.sqrt(denom))


def compute_fold_curve(pred_path: Path, fold: int) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    if "label" not in df.columns or "forecASD" not in df.columns:
        raise ValueError(f"{pred_path} must contain label and forecASD columns")

    y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    s = pd.to_numeric(df["forecASD"], errors="coerce").fillna(0.0).to_numpy()
    order = np.argsort(-s)
    y_sorted = y[order]
    n = int(len(y_sorted))
    total_pos = int(y_sorted.sum())
    total_neg = int(n - total_pos)

    rows = []
    for pct in PCTS:
        n_pred_pos = int(math.ceil(float(pct) * n))
        n_pred_pos = min(max(n_pred_pos, 1), n)
        pred_pos = y_sorted[:n_pred_pos]

        tp = int(pred_pos.sum())
        fp = int(n_pred_pos - tp)
        fn = int(total_pos - tp)
        tn = int(total_neg - fp)
        mcc = matthews_corrcoef_from_counts(tp, fp, tn, fn)

        rows.append({
            "fold": fold,
            "rank_percentage": float(pct),
            "n_test": n,
            "n_pred_pos": n_pred_pos,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "mcc": mcc,
        })
    return pd.DataFrame(rows)


def summarise_curve(curve_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, g in curve_df.groupby("fold", sort=True):
        g = g.sort_values("rank_percentage")
        x = g["rank_percentage"].to_numpy(dtype=float)
        y = g["mcc"].to_numpy(dtype=float)
        # Normalised AUC keeps the value on the natural MCC scale [-1, 1].
        auc = float(np.trapz(y, x) / max(float(x[-1] - x[0]), 1e-12))
        row = {
            "fold": int(fold),
            "mcc_auc": auc,
            "mcc_mean": float(np.mean(y)),
            "mcc_max": float(np.max(y)),
        }
        for name, pct in AT_PCTS.items():
            hit = g.loc[np.isclose(g["rank_percentage"], pct), "mcc"]
            row[name] = float(hit.iloc[0]) if not hit.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def append_fold_metrics(out_dir: Path, fold_summary: pd.DataFrame) -> None:
    path = out_dir / "cv_fold_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "fold" not in df.columns:
        return

    add_cols = [c for c in fold_summary.columns if c != "fold"]
    df = df.drop(columns=[c for c in add_cols if c in df.columns], errors="ignore")
    merged = df.merge(fold_summary, on="fold", how="left")
    merged.to_csv(path, index=False)


def append_summary_metrics(out_dir: Path, fold_summary: pd.DataFrame) -> None:
    path = out_dir / "cv_metrics_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    metric_names = [c for c in fold_summary.columns if c != "fold"]
    df = df[~df["metric"].isin(metric_names)].copy()

    rows = []
    for metric in metric_names:
        vals = pd.to_numeric(fold_summary[metric], errors="coerce")
        rows.append({
            "metric": metric,
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals, ddof=1)) if vals.notna().sum() > 1 else 0.0,
        })
    pd.concat([df, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


def process_output_dir(out_dir: Path) -> bool:
    fold_paths = sorted(out_dir.glob("fold_*/test_predictions.csv"))
    if not fold_paths:
        return False

    curves = []
    for pred_path in fold_paths:
        try:
            fold = int(pred_path.parent.name.split("_")[-1])
        except ValueError:
            continue
        curves.append(compute_fold_curve(pred_path, fold))

    if not curves:
        return False

    curve_df = pd.concat(curves, ignore_index=True)
    fold_summary = summarise_curve(curve_df)

    curve_df.to_csv(out_dir / "mcc_rank_threshold_curve.csv", index=False)
    fold_summary.to_csv(out_dir / "mcc_rank_threshold_fold_summary.csv", index=False)
    append_fold_metrics(out_dir, fold_summary)
    append_summary_metrics(out_dir, fold_summary)
    return True


def process_run_dirs() -> None:
    for i in range(1, 6):
        run_dir = ROOT / f"run_{i}"
        if not run_dir.exists():
            continue
        for out_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.endswith("_outputs")):
            if process_output_dir(out_dir):
                print(f"[OK] {out_dir.relative_to(ROOT)}")


def process_ablation_bce() -> None:
    base = ROOT / "ablation_bce"
    if not base.exists():
        return
    for out_dir in sorted(base.glob("w_bce_*/run_*/deep_motifs_outputs")):
        if process_output_dir(out_dir):
            print(f"[OK] {out_dir.relative_to(ROOT)}")


def average_metric_rows(paths: list[Path], metrics: list[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if p.exists():
            df = pd.read_csv(p).set_index("metric")
            dfs.append(df)
    rows = []
    for metric in metrics:
        means = np.array([df.loc[metric, "mean"] if metric in df.index else np.nan for df in dfs], dtype=float)
        stds = np.array([df.loc[metric, "std"] if metric in df.index else np.nan for df in dfs], dtype=float)
        rows.append({
            "metric": metric,
            "mean": float(np.nanmean(means)),
            "std": float(np.nanmean(stds)),
            "std_across_runs": float(np.nanstd(means, ddof=1)) if np.sum(~np.isnan(means)) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def update_mean_run() -> None:
    mean_root = ROOT / "mean_run"
    if not mean_root.exists():
        return

    metrics = ["mcc_auc", "mcc_mean", "mcc_max", *AT_PCTS.keys()]
    for mean_out in sorted(p for p in mean_root.iterdir() if p.is_dir() and p.name.endswith("_outputs")):
        algo = mean_out.name
        run_summary_paths = [ROOT / f"run_{i}" / algo / "cv_metrics_summary.csv" for i in range(1, 6)]
        if not any(p.exists() for p in run_summary_paths):
            continue

        metric_rows = average_metric_rows(run_summary_paths, metrics)
        summary_path = mean_out / "cv_metrics_summary.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df = df[~df["metric"].isin(metrics)].copy()
            pd.concat([df, metric_rows], ignore_index=True).to_csv(summary_path, index=False)

        curve_paths = [ROOT / f"run_{i}" / algo / "mcc_rank_threshold_curve.csv" for i in range(1, 6)]
        curves = [pd.read_csv(p).assign(run=i + 1) for i, p in enumerate(curve_paths) if p.exists()]
        if curves:
            curve_all = pd.concat(curves, ignore_index=True)
            curve_mean = (
                curve_all.groupby("rank_percentage", as_index=False)["mcc"]
                .agg(mean="mean", std="std")
                .rename(columns={"mean": "mcc_mean", "std": "mcc_std"})
            )
            curve_mean.to_csv(mean_out / "mcc_rank_threshold_curve.csv", index=False)
            metric_rows.to_csv(mean_out / "mcc_rank_threshold_summary.csv", index=False)
            print(f"[OK] {mean_out.relative_to(ROOT)}")


def main() -> None:
    process_run_dirs()
    process_ablation_bce()
    update_mean_run()


if __name__ == "__main__":
    main()

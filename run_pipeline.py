#!/usr/bin/env python3
"""Cross-platform benchmark pipeline driver (Windows / macOS / Linux).

Python port of run_multi.bat. Runs the paper-reproducibility pipeline for one
or more negative-sampling seeds:

    forecASD -> xgb -> rf -> svm -> deepgbm -> sai -> tabnet -> ftt -> cnn -> gcn
             -> deep_motifs

followed by the averaging / MCC post-processing steps. Results land in
run_1/ .. run_5/ (under --project-root) and are averaged into mean_run/.

Because everything is dispatched as ``python -m ...``, this single file runs
identically on any OS with a working Python + the project dependencies.

Examples
--------
    # All 5 runs + post-processing (default)
    python run_pipeline.py

    # Only run_1 (seed 42)
    python run_pipeline.py 1

    # Print the plan without executing anything
    python run_pipeline.py --dry-run

    # Skip forecASD and benchmark the models on a fixed label set
    python run_pipeline.py 1 --labels-dir ext_data/labels --skip-postprocess

    # Mirror what the batch file logs
    python run_pipeline.py --log run_pipeline.log
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Directory containing this script == repo root (has experiments/ + deep_motifs/).
# Subprocesses run with this as the working directory so `python -m experiments.*`
# and `python -m deep_motifs` always resolve, regardless of --project-root.
REPO_ROOT = Path(__file__).resolve().parent

# (run_id, negative-sampling seed) — identical to run_multi.bat.
RUNS = [(1, 42), (2, 123), (3, 456), (4, 789), (5, 1024)]

# Benchmark models run between forecASD and deep_motifs, in order.
MODELS = ["xgb", "rf", "sv", "deepgbm", "sai", "tab", "ftt", "cnn", "gcn"]

# deep_motifs uses the final Bayesian-guided PU defaults from the paper.
# (--prior-model xgb because lstm.py is not shipped.)
DEEP_MOTIFS_ARGS = [
    "--prior-model", "xgb",
    "--fusion-mode", "rrf",
    "--ppr-alpha", "1.0",
    "--pu-class-prior", "0.06",
    "--prior-uncertainty-delta", "0.05",
    "--prior-weight-floor", "0.10",
    "--prior-guided-calibration", "rank",
]


class Tee:
    """Write to stdout and, optionally, a log file at the same time."""

    def __init__(self, log_path: Path | None):
        self.file = open(log_path, "w", encoding="utf-8") if log_path else None

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        if self.file:
            self.file.write(text)
            self.file.flush()

    def close(self) -> None:
        if self.file:
            self.file.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run", nargs="?", type=int, choices=[r[0] for r in RUNS], default=None,
        help="Run only this seed (1-5). Omit to run all 5 and post-process.",
    )
    parser.add_argument(
        "--project-root", type=str, default=str(REPO_ROOT),
        help="Directory containing ext_data/ and where run_N/ outputs are written "
             "(default: this script's directory).",
    )
    parser.add_argument(
        "--python", type=str, default=sys.executable,
        help="Python interpreter used for each step (default: this interpreter).",
    )
    parser.add_argument(
        "--target-neg-ratio", type=float, default=1.0,
        help="Target negative:positive ratio passed to forecASD (default: 1.0).",
    )
    parser.add_argument(
        "--labels-dir", type=str, default=None,
        help="If set, skip forecASD and use this label directory for every model. "
             "Enables benchmarking on a fixed label set (e.g. ext_data/labels).",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated subset of models to run "
             f"(default: all -> {','.join(MODELS)},deep_motifs).",
    )
    parser.add_argument(
        "--skip-postprocess", action="store_true",
        help="Skip average_runs and add_mcc_auc_metrics.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run steps even if their outputs already exist.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would run without executing them.",
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Also write all output to this log file (e.g. run_pipeline.log).",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    target_ratio = args.target_neg_ratio
    log = Tee(Path(args.log).resolve() if args.log else None)

    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        requested = MODELS + ["deep_motifs"]
    model_steps = [m for m in requested if m in MODELS]
    run_deep = "deep_motifs" in requested

    runs = [r for r in RUNS if r[0] == args.run] if args.run else RUNS

    def emit(msg: str = "") -> None:
        log.write(msg + "\n")

    def run_cmd(cmd: list[str], step: str) -> bool:
        """Run one step. Return True on success (or dry-run)."""
        emit(f"[RUN] {step}")
        emit("      " + " ".join(cmd))
        if args.dry_run:
            emit(f"[OK] {step} (dry-run)")
            return True
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
        proc.wait()
        if proc.returncode != 0:
            emit(f"[FAILED] {step} (exit {proc.returncode})")
            return False
        emit(f"[OK] {step}")
        return True

    def forecasd_ratio_ok(report_file: Path) -> bool:
        try:
            data = json.loads(report_file.read_text(encoding="utf-8"))
            achieved = float(data.get("achieved_neg_ratio", -999.0))
            return abs(achieved - target_ratio) < 1e-9
        except Exception:
            return False

    py = args.python
    proj = str(project_root)

    try:
        for run_id, seed in runs:
            run_dir = project_root / f"run_{run_id}"
            emit("")
            emit("=" * 60)
            emit(f" [Run {run_id}] neg-random-state={seed}")
            emit("=" * 60)
            if not args.dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)

            forecasd_reran = False

            # --- forecASD (label generation), unless a fixed labels dir is given.
            if args.labels_dir:
                labels_dir = Path(args.labels_dir).resolve()
                emit(f"[SKIP] forecASD: using fixed labels dir {labels_dir}")
            else:
                labels_dir = run_dir / "forecasd_outputs"
                done = labels_dir / "all_labels_used.csv"
                report = labels_dir / "neg_selection_report.json"
                if not args.force and done.exists() and report.exists() \
                        and forecasd_ratio_ok(report):
                    emit(f"[SKIP] Run {run_id} forecasd already complete: {done}")
                else:
                    ok = run_cmd(
                        [py, "-m", "experiments.forecasd",
                         "--project-root", proj,
                         "--neg-random-state", str(seed),
                         "--target-neg-ratio", str(target_ratio),
                         "--output-dir", f"run_{run_id}/forecasd_outputs"],
                        f"Run {run_id} forecasd",
                    )
                    if not ok:
                        return 1
                    forecasd_reran = True

            labels_arg = str(labels_dir)

            # --- Benchmark models.
            for model in model_steps:
                done = run_dir / f"{model}_outputs" / "cv_metrics_summary.csv"
                if not args.force and done.exists() and not forecasd_reran:
                    emit(f"[SKIP] Run {run_id} {model} already complete: {done}")
                    continue
                ok = run_cmd(
                    [py, "-m", f"experiments.{model}",
                     "--project-root", proj,
                     "--labels-dir", labels_arg,
                     "--output-dir", f"run_{run_id}/{model}_outputs"],
                    f"Run {run_id} {model}",
                )
                if not ok:
                    return 1

            # --- Deep-MOTIFs.
            if run_deep:
                done = (run_dir / "deep_motifs_outputs"
                        / "cv_metrics_summary_global_threshold.csv")
                if not args.force and done.exists() and not forecasd_reran:
                    emit(f"[SKIP] Run {run_id} deep_motifs already complete: {done}")
                else:
                    ok = run_cmd(
                        [py, "-m", "deep_motifs",
                         "--project-root", proj,
                         "--labels-dir", labels_arg,
                         "--output-dir", f"run_{run_id}/deep_motifs_outputs",
                         *DEEP_MOTIFS_ARGS],
                        f"Run {run_id} deep_motifs",
                    )
                    if not ok:
                        return 1

            emit(f"[Run {run_id} done]")

        # --- Post-processing (only meaningful for a full multi-run sweep).
        if not args.skip_postprocess and not args.run:
            emit("")
            emit("=" * 60)
            emit(" Post-processing run_1..run_5")
            emit("=" * 60)
            for mod in ("experiments.average_runs", "experiments.add_mcc_auc_metrics"):
                if not run_cmd([py, "-m", mod, "--project-root", proj], mod):
                    return 1

        emit("")
        emit("=" * 60)
        emit(" Pipeline complete.")
        emit("=" * 60)
        return 0
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())

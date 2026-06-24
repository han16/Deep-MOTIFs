# Deep-MOTIFs

**Deep Multi-Omics Transformer with Integrated Features and Scores**

A Bayesian-guided positive-unlabeled (PU) deep learning framework for genome-wide ASD risk-gene prioritization. Integrates genetic, transcriptomic, and protein-interaction evidence using a multi-view Transformer with empirical-Bayes responsibility weighting and Reciprocal Rank Fusion.

---

## Requirements

- Python 3.9+
- GPU optional (CUDA 11.8+ recommended for full training runs)

**Python dependencies:**
```
numpy, pandas, scikit-learn, scipy, networkx, xgboost, statsmodels, torch
```

---

## Installation

```bash
git clone https://github.com/han16/Deep-MOTIFs.git
cd Deep-MOTIFs

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## External Data

The pipeline requires reference files in an `ext_data/` directory under your project root. These files are not included in the repository due to size.

| File | Source |
|------|--------|
| `9606.protein.links.v10.txt.gz` | [STRING v10](https://string-db.org/cgi/download) — human protein network |
| `composite_table.csv` | Composite genetic/functional feature table (see paper) |
| `tada_new.csv` | TADA gene scores |
| `jack_fu_gene_info(in).csv` | Additional gene annotations |
| BrainSpan expression files | [BrainSpan Atlas](https://www.brainspan.org/static/download.html) |

Place all files under:
```
<project-root>/
└── ext_data/
    ├── 9606.protein.links.v10.txt.gz
    ├── composite_table.csv
    ├── tada_new.csv
    ├── jack_fu_gene_info(in).csv
    └── ...
```

---

## Usage

### Basic run

```bash
python -m deep_motifs --project-root C:\path\to\your\data
```

Output is written to `<project-root>/deep_motifs_v4_outputs/`.

### All options

```bash
python -m deep_motifs --help
```

### Common options

```bash
python -m deep_motifs \
  --project-root C:\path\to\your\data \
  --output-dir my_run \
  --prior-model xgb \          # empirical prior: 'xgb' (recommended), 'none', or 'lstm'*
  --fusion-mode rrf \          # score fusion: 'rrf' (recommended), 'fixed', or 'pu'
  --pu-class-prior 0.06 \      # PU class prior pi (paper default)
  --prior-uncertainty-delta 0.05 \
  --prior-weight-floor 0.10 \
  --n-splits 5 \               # CV folds
  --device auto                # 'auto', 'cpu', or 'cuda'
```

> **Note:** `--prior-model lstm` requires `lstm.py` (not included in this repository). Use `--prior-model xgb` as an equivalent alternative.

---

## Output

```
<project-root>/<output-dir>/
├── cv_metrics_summary.csv                      # Mean ± std across CV folds (per-fold threshold)
├── cv_metrics_summary_global_threshold.csv     # Same, using global OOF threshold
├── full_scores_summary.csv                     # Genome-wide risk scores for all unlabeled genes
├── all_labels_used.csv
├── fold_infos_summary.csv
└── fold_1/ ... fold_N/
    ├── test_predictions.csv                    # Held-out predictions for this fold
    ├── all_gene_component_scores.csv           # Per-gene scores for every model component
    ├── full_scores.csv
    ├── fold_info.json
    └── xgb_empirical_prior_scores.csv
```

**The key output for downstream analysis is `full_scores_summary.csv`** — it contains the averaged genome-wide ASD risk score (`forecASD`) for every gene not in the labeled set, ranked from highest to lowest risk.

A `cache/` directory is created at `<project-root>/cache/` and stores prebuilt STRING graph files to speed up reruns.

---

## Paper Reproducibility

The full benchmark pipeline (all 5 runs × 10 models) used in the paper is in [`run_multi.bat`](run_multi.bat). Before running it:

1. Open `run_multi.bat` and update two lines at the top:
   ```bat
   set "PROJECT=C:\path\to\your\data"
   set "PYTHON=C:\path\to\.venv\Scripts\python.exe"
   ```

2. Change `--prior-model lstm` to `--prior-model xgb` on line 216 (lstm.py is not included).

3. Run from the project root:
   ```bat
   run_multi.bat        # all 5 runs
   run_multi.bat 1      # only run_1
   ```

The script runs these models on the same labeled gene set:
`forecasd → xgb → rf → svm → deepgbm → sai → tabnet → ftt → cnn → gcn → deep_motifs`

Results land in `run_1/` … `run_5/` and are averaged into `mean_run/`.

---

## Package Structure

```
deep_motifs/
├── __main__.py          # python -m deep_motifs entry point
├── cli.py               # argument parsing and main()
├── cv.py                # cross-validation driver (run_pu)
├── training.py          # fit_deep_motifs_and_export, training loop
├── models.py            # DeepMOTIFs transformer, BrainSpanEncoder, tokenizers
├── pretrain.py          # masked-feature reconstruction pretraining
├── priors.py            # XGB / LSTM fold-local empirical prior estimators
├── losses.py            # nnPU, pairwise ranking, weighted losses
├── fusion.py            # RRF, alpha-weighted, asymmetric fusion
├── ppr.py               # Personalized PageRank score propagation
├── graph.py             # STRING graph loading, GCN aggregation
├── calibration.py       # threshold search, score remapping
├── features.py          # view frame construction, standardization
├── poly_features.py     # XGB-guided polynomial feature expansion
├── augmentation.py      # feature masking / corruption
├── noise.py             # feature and label noise injection
└── reproducibility.py   # seed setting, device resolution
```

---

## Citation


```

---

## License

[Add your license here]

from __future__ import annotations

import argparse
from pathlib import Path

from .xgb import (
    augment_composite_with_tada,
    build_brainspan_matrix,
    build_string_graph,
    ensure_exists,
    load_composite_table,
    load_labels,
)

from .pipeline import run_pu
from .utils import resolve_device

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep-MOTIFs — v25 + global OOF threshold calibration.")
    p.add_argument("--project-root",  type=str, required=True)
    p.add_argument("--labels-dir",    type=str, default=None)
    p.add_argument("--output-dir",    type=str, default="deep_motifs_outputs")
    p.add_argument("--tada-filename",    type=str, default="tada_new.csv")
    p.add_argument("--jack-filename",    type=str, default="jack_fu_gene_info(in).csv")
    p.add_argument("--n-splits",      type=int, default=5)
    p.add_argument("--random-state",  type=int, default=42)
    p.add_argument("--string-mode",   type=str, default="anchor",
                   choices=["anchor", "graph"])
    p.add_argument("--max-string-anchors", type=int, default=256)
    p.add_argument("--device",        type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--force-rebuild-brainspan",      action="store_true")
    p.add_argument("--force-rebuild-string",         action="store_true")
    p.add_argument("--force-rebuild-graph-features", action="store_true")
    p.add_argument("--force-rebuild-xgb-oof",        action="store_true")

    # Architecture  (meta_token_count removed 鈥?MetaMLP always gives 1 token)
    p.add_argument("--token-dim",          type=int,   default=64)
    p.add_argument("--bs-n-regions",       type=int,   default=16)
    p.add_argument("--bs-n-timepoints",    type=int,   default=50)
    p.add_argument("--str-token-count",    type=int,   default=6)
    p.add_argument("--transformer-heads",  type=int,   default=4)
    p.add_argument("--transformer-layers", type=int,   default=2)
    p.add_argument("--dropout",            type=float, default=0.3)

    # Training
    p.add_argument("--epochs",             type=int,   default=150)
    p.add_argument("--batch-size",         type=int,   default=128)
    p.add_argument("--learning-rate",      type=float, default=1e-4)
    p.add_argument("--weight-decay",       type=float, default=5e-3)
    p.add_argument("--patience",           type=int,   default=30)
    p.add_argument("--warmup-epochs",      type=int,   default=10)
    p.add_argument("--early-stop-metric",  type=str,   default="pr_auc",
                   choices=["pr_auc", "recall_at_k", "loss"])
    p.add_argument("--early-stop-k",       type=int,   default=50)

    # Augmentation
    p.add_argument("--augment-factor",  type=int,   default=3)
    p.add_argument("--augment-scale",   type=float, default=0.4)
    p.add_argument("--mask-rate-meta",  type=float, default=0.1)
    p.add_argument("--mask-rate-bs",    type=float, default=0.15)
    p.add_argument("--mask-rate-str",   type=float, default=0.15)
    p.add_argument("--noise-std",       type=float, default=0.03)

    # Loss weights  (v2: BCE removed from fine-tune; nnPU is main loss)
    p.add_argument("--w-bce",   type=float, default=0.0)
    p.add_argument("--w-pu",    type=float, default=1.0)
    p.add_argument("--w-rank",  type=float, default=0.5)
    p.add_argument("--w-oc",    type=float, default=0.05)
    p.add_argument("--w-graph", type=float, default=0.05)
    p.add_argument("--w-cons",  type=float, default=0.05)

    # Graph
    p.add_argument("--graph-top-k",   type=int, default=3)
    p.add_argument("--gcn-n-layers",  type=int, default=2,
                   help="Number of GCN aggregation layers (default: 2).")

    # XGBoost OOF
    p.add_argument("--use-xgb-feature",  dest="use_xgb_feature",
                   action="store_true",  default=True)
    p.add_argument("--no-xgb-feature",   dest="use_xgb_feature",
                   action="store_false")
    p.add_argument("--xgb-n-estimators",      type=int,   default=500,
                   help="Number of XGBoost trees (v18 default=500).")
    p.add_argument("--xgb-max-depth",         type=int,   default=4,
                   help="XGBoost max_depth (v18 default=4, was 6 in v8).")
    p.add_argument("--xgb-min-child-weight",  type=int,   default=5,
                   help="XGBoost min_child_weight (v18 default=5, was 1 in v8).")
    p.add_argument("--xgb-reg-alpha",         type=float, default=0.1,
                   help="XGBoost L1 regularisation (v18 default=0.1).")
    p.add_argument("--xgb-gamma",             type=float, default=0.1,
                   help="XGBoost gamma min-split-loss (v18 default=0.1).")

    # Speed
    p.add_argument("--center-update-interval", type=int, default=5)
    p.add_argument("--use-torch-compile",  action="store_true", default=False)

    # v7: Checkpoint Averaging
    p.add_argument("--ckpt-avg-k", type=int, default=5,
                   help="Average the last K improved checkpoints (1 = use single best, no averaging)")

    # v8: Fusion mode
    p.add_argument("--fusion-mode", type=str, default="rrf",
                   choices=["fixed", "rrf"],
                   help="Score fusion strategy: 'fixed' = 0.5脳XGB+0.5脳PU, 'rrf' = Reciprocal Rank Fusion")
    p.add_argument("--rrf-k", type=int, default=60,
                   help="RRF constant k (only used when --fusion-mode=rrf). "
                        "Classic value=60. Smaller 鈫?top ranks dominate more.")

    # v16: Personalized PageRank propagation
    p.add_argument("--ppr-alpha", type=float, default=0.5,
                   help="PPR restart probability. 1.0 = no propagation (same as v8). "
                        "Lower values spread scores more through the STRING network. "
                        "Default=0.5.")
    p.add_argument("--ppr-n-iter", type=int, default=30,
                   help="Max power-iteration steps for PPR (default=30).")
    p.add_argument("--ppr-min-edge-weight", type=float, default=0.5,
                   help="Minimum normalised STRING edge weight to use in propagation. "
                        "0.5 corresponds to STRING confidence score > 700 "
                        "(when base threshold=400). Default=0.5.")
    p.add_argument("--ppr-fusion-weight", type=float, default=0.7,
                   help="v20: PPR multiplier in asymmetric RRF: "
                        "score = 1/(k+rank_model) + ppr_w/(k+rank_ppr). "
                        "1.0 = symmetric RRF (like v18); 0.7 = model ~59%% PPR ~41%%. "
                        "Lower values preserve model precision; higher values boost recall. "
                        "Default=0.7.")

    # v25: XGB-guided polynomial feature expansion
    p.add_argument("--poly-top-k", type=int, default=6,
                   help="v25: top-K meta features by XGB gain importance for polynomial "
                        "expansion (pairwise products + squares). 0 = disabled.")

    # Ablation flags
    p.add_argument("--ablate-string",    action="store_true",
                   help="Ablation: replace all STRING features with zeros (removes graph info).")
    p.add_argument("--ablate-brainspan", action="store_true",
                   help="Ablation: replace all BrainSpan features with zeros (removes temporal info).")
    # Noise robustness
    p.add_argument("--noise-type", type=str, default="none", choices=["none", "gaussian", "dropout"],
                   help="Feature noise type applied to training data only (default: none)")
    p.add_argument("--noise-level", type=float, default=0.0,
                   help="Noise level: std multiplier for gaussian, drop rate for dropout (default: 0.0)")
    p.add_argument("--label-flip-rate", type=float, default=0.0,
                   help="Fraction of negative training labels flipped to positive (default: 0.0)")

    # v6: Masked pre-training
    p.add_argument("--pretrain-epochs",    type=int,   default=50,
                   help="Number of masked feature reconstruction pre-training epochs (0 = skip)")
    p.add_argument("--pretrain-lr",        type=float, default=1e-3,
                   help="Learning rate for pre-training optimizer")
    p.add_argument("--pretrain-mask-rate", type=float, default=0.30,
                   help="Fraction of meta/BrainSpan features to mask during pre-training")
    p.add_argument("--w-pretrain-pu", type=float, default=0.3,
                   help="Weight of nnPU loss during pretrain (v2)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    print(f"[INFO] Using project root: {project_root}")
    data_dir = project_root / "data"
    ensure_exists(data_dir, "data directory")

    labels_dir = (
    Path(args.labels_dir).resolve()
    if args.labels_dir
    else data_dir / "labels"
    )
    output_dir = project_root / args.output_dir

    print("[INFO] Loading composite table...")
    meta_df = load_composite_table(data_dir)

    tada_path = data_dir / args.tada_filename
    jack_path = data_dir / args.jack_filename
    
    print(tada_path)
    print(jack_path)

    ensure_exists(tada_path, args.tada_filename)
    ensure_exists(jack_path, args.jack_filename)
    print("[INFO] Augmenting composite table with tada_new features...")
    meta_df = augment_composite_with_tada(meta_df, tada_path, jack_path)
    print(f"[INFO] Augmented composite table shape: {meta_df.shape}")

    print("[INFO] Loading labels...")
    labels_df = load_labels(labels_dir)
    print("[INFO] Building STRING graph...")
    G = build_string_graph(data_dir, force_rebuild=args.force_rebuild_string)
    print(f"[INFO] STRING graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("[INFO] Building BrainSpan matrix...")
    brainspan_df = build_brainspan_matrix(
        ext_data_dir=data_dir,
        target_proteins=set(meta_df.index.astype(str)),
        force_rebuild=args.force_rebuild_brainspan,
    )
    print(f"[INFO] BrainSpan shape: {brainspan_df.shape}")

    device = resolve_device(args.device)
    fusion_info = (
        f"rrf(k={args.rrf_k})" if args.fusion_mode == "rrf" else "fixed(alpha=0.5)"
    )
    print(
        f"[INFO] Deep-MOTIFs v20 start 鈥?device={device}  "
        f"token_dim={args.token_dim}  "
        f"ckpt_avg_k={args.ckpt_avg_k}  "
        f"early_stop={args.early_stop_metric}  "
        f"pretrain_epochs={args.pretrain_epochs}  "
        f"fusion={fusion_info}  "
        f"ppr_alpha={args.ppr_alpha}  "
        f"use_xgb={args.use_xgb_feature}"
    )

    run_pu(
        labels_df=labels_df,
        meta_df=meta_df,
        brainspan_df=brainspan_df,
        G=G,
        data_dir=data_dir,
        output_dir=output_dir,
        string_mode=args.string_mode,
        max_string_anchors=args.max_string_anchors,
        n_splits=args.n_splits,
        random_state=args.random_state,
        force_rebuild_graph_features=args.force_rebuild_graph_features,
        device=device,
        token_dim=args.token_dim,
        bs_n_regions=args.bs_n_regions,
        bs_n_timepoints=args.bs_n_timepoints,
        str_token_count=args.str_token_count,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        early_stop_metric=args.early_stop_metric,
        early_stop_k=args.early_stop_k,
        augment_factor=args.augment_factor,
        augment_scale=args.augment_scale,
        mask_rate_meta=args.mask_rate_meta,
        mask_rate_bs=args.mask_rate_bs,
        mask_rate_str=args.mask_rate_str,
        noise_std=args.noise_std,
        w_bce=args.w_bce,
        w_pu=args.w_pu,
        w_rank=args.w_rank,
        w_oc=args.w_oc,
        w_graph=args.w_graph,
        w_cons=args.w_cons,
        graph_top_k=args.graph_top_k,
        use_xgb_feature=args.use_xgb_feature,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_min_child_weight=args.xgb_min_child_weight,
        xgb_reg_alpha=args.xgb_reg_alpha,
        xgb_gamma=args.xgb_gamma,
        gcn_n_layers=args.gcn_n_layers,
        warmup_epochs=args.warmup_epochs,
        center_update_interval=args.center_update_interval,
        use_torch_compile=args.use_torch_compile,
        force_rebuild_xgb_oof=args.force_rebuild_xgb_oof,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_lr=args.pretrain_lr,
        pretrain_mask_rate=args.pretrain_mask_rate,
        w_pretrain_pu=args.w_pretrain_pu,
        ckpt_avg_k=args.ckpt_avg_k,
        fusion_mode=args.fusion_mode,
        rrf_k=args.rrf_k,
        ppr_alpha=args.ppr_alpha,
        ppr_n_iter=args.ppr_n_iter,
        ppr_min_edge_weight=args.ppr_min_edge_weight,
        ppr_fusion_weight=args.ppr_fusion_weight,
        poly_top_k=args.poly_top_k,
        ablate_string=args.ablate_string,
        ablate_brainspan=args.ablate_brainspan,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
        label_flip_rate=args.label_flip_rate,
    )
    print(f"[DONE] v26 results saved to: {output_dir}")


if __name__ == "__main__":
    main()
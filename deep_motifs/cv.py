from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from .xgb import build_fold_string_feature_matrix
from .xgb import compute_graph_features
from .xgb import evaluate_predictions
from .label_noise_utils import apply_training_label_budget
from .label_noise_utils import apply_training_label_perturbation

from .noise import _apply_label_noise
from .features import build_view_frames
from .features import standardize_fit_and_all
from .fusion import fuse_scores
from .fusion import rrf_fuse_scores
from .fusion import asymmetric_rrf_fuse
from .ppr import compute_ppr_from_seeds
from .poly_features import poly_expand_meta
from .losses import nanmean_std
from .calibration import find_best_threshold_by_f1
from .calibration import remap_score_with_threshold
from .graph import build_weighted_string_graph
from .models import DeepMOTIFs
from .pretrain import pretrain_encoder
from .pretrain import compute_pretrain_meta_importance
from .priors import compute_xgb_fold_local_prior_scores
from .priors import compute_lstm_fold_local_prior_scores
from .training import fit_deep_motifs_and_export


def run_pu(
    labels_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    brainspan_df: pd.DataFrame,
    G,
    ext_data_dir: Path,
    output_dir: Path,
    string_mode: str,
    max_string_anchors: int,
    n_splits: int,
    random_state: int,
    force_rebuild_graph_features: bool,
    device: torch.device,
    token_dim: int,
    bs_n_regions: int,
    bs_n_timepoints: int,
    str_token_count: int,
    transformer_heads: int,
    transformer_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    early_stop_metric: str,
    early_stop_k: int,
    augment_factor: int,
    augment_scale: float,
    mask_rate_meta: float,
    mask_rate_bs: float,
    mask_rate_str: float,
    noise_std: float,
    w_sup: float,
    w_bce: float,
    w_pu: float,
    w_rank: float,
    w_oc: float,
    w_graph: float,
    w_cons: float,
    graph_top_k: int,
    prior_model: str,
    use_xgb_feature: bool,
    xgb_n_estimators: int,
    gcn_n_layers: int,
    warmup_epochs: int,
    center_update_interval: int,
    use_torch_compile: bool,
    force_rebuild_xgb_oof: bool,
    xgb_max_depth: int = 4,
    xgb_min_child_weight: int = 5,
    xgb_reg_alpha: float = 0.1,
    xgb_gamma: float = 0.1,
    pretrain_epochs: int = 50,
    pretrain_lr: float = 1e-3,
    pretrain_mask_rate: float = 0.30,
    w_pretrain_pu: float = 0.3,
    pu_class_prior: float | None = None,
    prior_guided_pu: bool = True,
    prior_guided_calibration: str = "rank",
    prior_weight_floor: float = 0.10,
    prior_uncertainty_delta: float = 0.0,
    ckpt_avg_k: int = 5,
    fusion_mode: str = "fixed",
    rrf_k: int = 60,
    ppr_alpha: float = 1.0,
    ppr_n_iter: int = 30,
    ppr_min_edge_weight: float = 0.5,
    ppr_fusion_weight: float = 0.7,
    poly_top_k: int = 0,          # deep_motifs2: disabled by default for stricter CV
    ablate_string: bool = False,
    ablate_brainspan: bool = False,
    noise_type: str = "none",
    noise_level: float = 0.0,
    label_flip_rate: float = 0.0,
    label_noise_mode: str = "none",
    label_noise_rate: float = 0.0,
    label_budget_positive_fraction: float = 1.0,
    label_budget_neg_ratio: float = 0.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_df = labels_df.reset_index(drop=True)

    target_ids = meta_df.index.astype(str).tolist()
    valid_ids  = set(meta_df.index) & set(G.nodes)
    labels_df  = labels_df[labels_df["id"].isin(valid_ids)].reset_index(drop=True)

    n_pos = int((labels_df["label"] == 1).sum())
    n_neg = int((labels_df["label"] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Only one class after filtering. n_pos={n_pos}, n_neg={n_neg}.")
    labels_df.to_csv(output_dir / "all_labels_used.csv", index=False)

    # Build the weighted STRING graph, used for the weighted GCN pre-aggregation
    print("[INFO] Building weighted STRING graph for GCN aggregation...")
    weighted_G = build_weighted_string_graph(
        ext_data_dir=ext_data_dir,
        score_threshold=400,
        cache_path=output_dir.parent / "cache" / "string_gene_graph_weighted.pkl",
    )
    print(
        f"[INFO] Weighted STRING graph: "
        f"{weighted_G.number_of_nodes()} nodes, {weighted_G.number_of_edges()} edges"
    )

    # STRING graph features (graph mode)
    string_graph_features: pd.DataFrame | None = None
    if string_mode == "graph":
        cache_path = output_dir.parent / "cache" / "string_gene_graph_features.pkl"
        string_graph_features = compute_graph_features(
            G=G, target_ids=target_ids,
            cache_path=cache_path, force_rebuild=force_rebuild_graph_features,
        )
        string_graph_features = (
            string_graph_features.reindex(meta_df.index)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(string_graph_features.median(numeric_only=True))
            .fillna(0.0)
        )

    rng = np.random.default_rng(random_state)

    prior_model = str(prior_model).lower()
    if use_xgb_feature and prior_model == "none":
        prior_model = "xgb"
    if prior_model not in {"none", "xgb", "lstm"}:
        raise ValueError(f"Unknown prior_model={prior_model}")
    if prior_model != "none":
        print(f"[INFO] Empirical prior will be fold-local ({prior_model}).")

    # v25: Pretrain reconstruction-error polynomial meta-feature expansion.
    # A temporary Deep-MOTIFs is pretrained (fully unsupervised) on ALL genes using
    # masked-feature reconstruction.  Per-feature ablation MSE ranks features with
    # ZERO label leakage — no labels are touched during importance computation.
    # The same top-K pairs/squares are used for polynomial expansion in every fold.
    top_meta_pairs:   list = []
    top_meta_squares: list = []
    if use_xgb_feature and poly_top_k > 0:
        print(
            "[WARN] --poly-top-k > 0 uses one global unsupervised pretrain "
            "importance pass. Use --poly-top-k 0 for the cleanest fold-local "
            "prior experiment."
        )
        poly_string_df = build_fold_string_feature_matrix(
            G=G, target_ids=target_ids,
            anchor_ids=labels_df["id"].tolist(),
            max_anchors=max_string_anchors,
        )
        oof_meta, oof_bs, oof_str = build_view_frames(
            meta_df=meta_df, brainspan_df=brainspan_df, string_df=poly_string_df,
        )
        if ablate_string:
            oof_str = pd.DataFrame(
                np.zeros(oof_str.shape, dtype=np.float32),
                index=oof_str.index, columns=oof_str.columns,
            )
        if ablate_brainspan:
            oof_bs = pd.DataFrame(
                np.zeros(oof_bs.shape, dtype=np.float32),
                index=oof_bs.index, columns=oof_bs.columns,
            )
        print(
            f"[INFO] v25: pretrain-MSE feature selection "
            f"(top-{poly_top_k}, zero label leakage)..."
        )
        # Reuse oof_meta / oof_bs already built above; standardise on all genes.
        _x_meta_pt = standardize_fit_and_all(
            oof_meta.to_numpy(dtype=np.float32),
            oof_meta.to_numpy(dtype=np.float32),
        )
        _x_bs_pt = standardize_fit_and_all(
            oof_bs.to_numpy(dtype=np.float32),
            oof_bs.to_numpy(dtype=np.float32),
        )
        _str_dim_pt = oof_str.shape[1]
        _meta_t_pt  = torch.from_numpy(_x_meta_pt).float()
        _bs_t_pt    = torch.from_numpy(_x_bs_pt).float()

        _tmp_model = DeepMOTIFs(
            meta_dim=_x_meta_pt.shape[1],
            bs_dim=_x_bs_pt.shape[1],
            str_dim=_str_dim_pt,
            token_dim=token_dim,
            bs_n_regions=bs_n_regions,
            bs_n_timepoints=bs_n_timepoints,
            str_token_count=str_token_count,
            n_heads=transformer_heads,
            n_layers=transformer_layers,
            dropout=dropout,
        ).to(device)

        # Keep this global feature-importance pass label-free. Fold-specific
        # model pretraining below may still use outer-train positives only.
        _pos_global_idx_pt = np.asarray([], dtype=np.int64)

        _meta_dec_pt, _ = pretrain_encoder(
            model=_tmp_model,
            meta_t=_meta_t_pt,
            bs_t=_bs_t_pt,
            str_dim=_str_dim_pt,
            device=device,
            pretrain_epochs=pretrain_epochs,
            pretrain_lr=pretrain_lr,
            pretrain_mask_rate=pretrain_mask_rate,
            batch_size=batch_size,
            pos_global_idx=_pos_global_idx_pt,
            w_pretrain_pu=0.0,
            progress_prefix="[PretrainImportance] ",
        )

        top_meta_pairs, top_meta_squares = compute_pretrain_meta_importance(
            model=_tmp_model,
            meta_decoder=_meta_dec_pt,
            meta_t=_meta_t_pt,
            bs_t=_bs_t_pt,
            str_dim=_str_dim_pt,
            device=device,
            meta_col_names=oof_meta.columns.tolist(),
            top_k=poly_top_k,
            batch_size=batch_size,
        )
        del _tmp_model, _meta_dec_pt, _meta_t_pt, _bs_t_pt

        n_new = len(top_meta_pairs) + len(top_meta_squares)
        print(f"[INFO] v25: top features (pretrain MSE): {top_meta_squares}")
        print(
            f"[INFO] v25: {len(top_meta_pairs)} cross-products + "
            f"{len(top_meta_squares)} squares = {n_new} new meta features"
        )

    skf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X_ids = labels_df["id"].values
    y     = labels_df["label"].values

    all_metrics:           list[dict[str, float]] = []
    full_scores_unlabeled: list[pd.DataFrame]     = []
    label_ids_set = set(labels_df["id"].astype(str))
    fold_infos: list[dict] = []

    # v26: global OOF threshold — collect raw (pre-remap) test scores across all folds
    oof_raw_scores: dict[str, float] = {}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_ids, y), start=1):
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df = labels_df.iloc[train_idx].copy().reset_index(drop=True)
        test_df  = labels_df.iloc[test_idx].copy().reset_index(drop=True)
        train_df, label_noise_info = apply_training_label_perturbation(
            train_df, label_noise_mode, label_noise_rate, random_state + fold_idx * 1009
        )
        train_df, label_budget_info = apply_training_label_budget(
            train_df, label_budget_positive_fraction, label_budget_neg_ratio, random_state + fold_idx * 2003
        )
        if label_flip_rate > 0.0:
            _y = train_df["label"].to_numpy(dtype=int)
            train_df["label"] = _apply_label_noise(_y, label_flip_rate, rng)
        train_df.to_csv(fold_dir / "train_labels.tsv", sep="\t", index=False)
        test_df.to_csv( fold_dir / "test_labels.tsv",  sep="\t", index=False)
        test_ids = set(test_df["id"].tolist())

        print(
            f"[INFO] Fold {fold_idx}/{n_splits}: "
            f"n_train={len(train_df)}  n_test={len(test_df)}  "
            f"n_pos_train={int(train_df['label'].sum())}  "
            f"n_neg_train={int((train_df['label']==0).sum())}"
        )

        if string_mode == "anchor":
            string_feature_df = build_fold_string_feature_matrix(
                G=G, target_ids=target_ids,
                anchor_ids=train_df["id"].tolist(),
                max_anchors=max_string_anchors,
            )
        else:
            if string_graph_features is None:
                raise ValueError("string_graph_features is not built")
            string_feature_df = string_graph_features

        meta_all, bs_all, str_all = build_view_frames(
            meta_df=meta_df, brainspan_df=brainspan_df, string_df=string_feature_df,
        )
        if ablate_string:
            str_all = pd.DataFrame(
                np.zeros(str_all.shape, dtype=np.float32),
                index=str_all.index, columns=str_all.columns,
            )
        if ablate_brainspan:
            bs_all = pd.DataFrame(
                np.zeros(bs_all.shape, dtype=np.float32),
                index=bs_all.index, columns=bs_all.columns,
            )

        # Feature noise: applied to training rows of each view
        if noise_type != "none" and noise_level > 0.0:
            _train_ids = train_df["id"].tolist()
            for _view in [meta_all, bs_all, str_all]:
                _arr = _apply_feature_noise(
                    _view.loc[_train_ids].to_numpy(dtype=np.float32),
                    noise_type, noise_level, rng,
                )
                _view.loc[_train_ids] = _arr

        # v25: expand meta_all with XGB-guided polynomial features
        if top_meta_pairs or top_meta_squares:
            meta_all = poly_expand_meta(meta_all, top_meta_pairs, top_meta_squares)

        fold_prior_scores: pd.Series | None = None
        if prior_model == "xgb":
            print(f"[Fold {fold_idx}] Fitting fold-local XGBoost empirical prior...")
            fold_prior_scores = compute_xgb_fold_local_prior_scores(
                train_df=train_df,
                meta_all=meta_all,
                bs_all=bs_all,
                str_all=str_all,
                n_splits=n_splits,
                random_state=random_state + fold_idx * 1009,
                n_estimators=xgb_n_estimators,
                xgb_max_depth=xgb_max_depth,
                xgb_min_child_weight=xgb_min_child_weight,
                xgb_reg_alpha=xgb_reg_alpha,
                xgb_gamma=xgb_gamma,
            )
            prior_file = "xgb_empirical_prior_scores.csv"
            prior_col = "xgb_empirical_prior_score"
        elif prior_model == "lstm":
            print(f"[Fold {fold_idx}] Fitting fold-local LSTM empirical prior...")
            fold_prior_scores = compute_lstm_fold_local_prior_scores(
                train_df=train_df,
                meta_all=meta_all,
                bs_all=bs_all,
                str_all=str_all,
                n_splits=n_splits,
                random_state=random_state + fold_idx * 1009,
                device=device,
                bs_n_regions=bs_n_regions,
                bs_n_timepoints=bs_n_timepoints,
                epochs=120,
                batch_size=batch_size,
                lr=5e-4,
                patience=20,
                hidden_size=64,
                num_layers=2,
                dropout=dropout,
            )
            prior_file = "lstm_empirical_prior_scores.csv"
            prior_col = "lstm_empirical_prior_score"
        else:
            prior_file = ""
            prior_col = ""

        if fold_prior_scores is not None:
            fold_prior_scores.to_csv(
                fold_dir / prior_file,
                header=[prior_col],
            )
            pos_train_ids = train_df.loc[train_df["label"] == 1, "id"].tolist()
            neg_train_ids = train_df.loc[train_df["label"] == 0, "id"].tolist()
            test_ids_list = test_df["id"].tolist()
            print(
                f"[Fold {fold_idx}] {prior_model.upper()} prior done. "
                f"train_pos_mean={fold_prior_scores.loc[pos_train_ids].mean():.4f}  "
                f"train_neg_mean={fold_prior_scores.loc[neg_train_ids].mean():.4f}  "
                f"test_mean={fold_prior_scores.loc[test_ids_list].mean():.4f}"
            )

        score_all, feat_all, fit_info = fit_deep_motifs_and_export(
            X_meta_all_raw=meta_all,
            X_bs_all_raw=bs_all,
            X_str_all_raw=str_all,
            ids_all=target_ids,
            train_df=train_df,
            test_ids=test_ids,
            G=G,
            random_state=random_state + fold_idx * 101,
            device=device,
            token_dim=token_dim,
            bs_n_regions=bs_n_regions,
            bs_n_timepoints=bs_n_timepoints,
            str_token_count=str_token_count,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            early_stop_metric=early_stop_metric,
            early_stop_k=early_stop_k,
            augment_factor=augment_factor,
            augment_scale=augment_scale,
            mask_rate_meta=mask_rate_meta,
            mask_rate_bs=mask_rate_bs,
            mask_rate_str=mask_rate_str,
            noise_std=noise_std,
            w_sup=w_sup,
            w_bce=w_bce,
            w_pu=w_pu,
            w_rank=w_rank,
            w_oc=w_oc,
            w_graph=w_graph,
            w_cons=w_cons,
            graph_top_k=graph_top_k,
            xgb_oof_scores=fold_prior_scores,
            weighted_G=weighted_G,
            gcn_n_layers=gcn_n_layers,
            warmup_epochs=warmup_epochs,
            center_update_interval=center_update_interval,
            use_torch_compile=use_torch_compile,
            pretrain_epochs=pretrain_epochs,
            pretrain_lr=pretrain_lr,
            pretrain_mask_rate=pretrain_mask_rate,
            w_pretrain_pu=w_pretrain_pu,
            pu_class_prior=pu_class_prior,
            prior_guided_pu=prior_guided_pu,
            prior_guided_calibration=prior_guided_calibration,
            prior_weight_floor=prior_weight_floor,
            prior_uncertainty_delta=prior_uncertainty_delta,
            ckpt_avg_k=ckpt_avg_k,
            head_rrf_k=rrf_k,
            progress_prefix=f"[Fold {fold_idx}][PU] ",
        )

        # v4: alpha is fixed at 0.5; no longer searched on the training set.
        # v8: added the fusion_mode="rrf" option, using Reciprocal Rank Fusion
        #     instead of linear weighting.
        # best_alpha = 0.5
        best_alpha = float("nan")
        best_fusion_auc = float("nan")
        if fold_prior_scores is not None:
            xgb_all = fold_prior_scores.reindex(score_all.index).fillna(0.5).to_numpy(dtype=float)
            pu_all  = score_all.to_numpy(dtype=float)

            if fusion_mode == "pu":
                print(
                    f"[Fold {fold_idx}] Fusion disabled: using PU posterior score only"
                )
                fused_all = pu_all
                best_alpha = -2.0   # sentinel: indicates PU-only ablation in CSV
            elif fusion_mode == "rrf":
                print(
                    f"[Fold {fold_idx}] 铻嶅悎: RRF (k={rrf_k})"
                )
                fused_all = rrf_fuse_scores(xgb_all, pu_all, k=rrf_k)
                best_alpha = -1.0   # sentinel: indicates RRF mode in CSV
            else:
                best_alpha = 0.5
                print(
                    f"[Fold {fold_idx}] 铻嶅悎: fixed alpha=0.50 (XGB鏉冮噸=50%, PU鏉冮噸=50%)"
                )
                fused_all = fuse_scores(xgb_all, pu_all, alpha=best_alpha)

            score_all = pd.Series(fused_all, index=score_all.index, dtype=float)

        # v17: Seeded PPR 鈥?propagate FROM known training positives through STRING
        # v18: save pre-PPR scores for threshold calibration
        # v20: confidence-weighted seeds (XGBoost OOF) + asymmetric PPR fusion weight
        score_all_pre_ppr = score_all.copy()
        ppr_scores: pd.Series | None = None
        if ppr_alpha < 1.0:
            train_pos_ids = train_df[train_df["label"] == 1]["id"].tolist()

            # v20: weight each seed by its XGBoost OOF confidence score
            # High-confidence positives contribute more to PPR 鈫?less noise propagation
            if fold_prior_scores is not None:
                raw_w   = {gid: float(fold_prior_scores.get(gid, 0.5)) for gid in train_pos_ids}
                total_w = max(sum(raw_w.values()), 1e-9)
                seed_w  = {k: v / total_w for k, v in raw_w.items()}
            else:
                seed_w  = None

            ppr_scores = compute_ppr_from_seeds(
                seed_ids=train_pos_ids,
                all_ids=list(score_all.index),
                G=weighted_G,
                alpha=ppr_alpha,
                n_iter=ppr_n_iter,
                min_edge_weight=ppr_min_edge_weight,
                seed_weights=seed_w,          # v20: confidence-weighted
            )
            # v20: asymmetric RRF (model ~59%, PPR ~41%) 鈥?preserves right-skewed distribution
            rrf_arr = score_all.to_numpy(dtype=float)
            ppr_arr = ppr_scores.reindex(score_all.index).fillna(0.0).to_numpy(dtype=float)
            final_arr = asymmetric_rrf_fuse(rrf_arr, ppr_arr, k=rrf_k, ppr_w=ppr_fusion_weight)
            score_all = pd.Series(final_arr, index=score_all.index, dtype=float)
            print(
                f"[Fold {fold_idx}] Seeded PPR done  "
                f"n_seeds={len(train_pos_ids)}  alpha={ppr_alpha}  "
                f"min_w={ppr_min_edge_weight}"
            )

        # v19: hybrid calibration -- positives use pre-PPR (avoids seed inflation),
        #      negatives use post-PPR (same distribution as the test set).
        # Training positives are PPR seeds, so their post-PPR score is ~1.0
        # (artificial inflation) -> restore them to the pre-PPR score.
        # use train-set scores to calibrate threshold consistently with test distribution
        train_ids_fold  = fit_info["train_ids"]
        train_labels    = np.asarray(fit_info["train_labels"], dtype=int)
        if ppr_alpha < 1.0:
            post_arr    = score_all.reindex(train_ids_fold).to_numpy(dtype=float)
            pre_arr     = score_all_pre_ppr.reindex(train_ids_fold).to_numpy(dtype=float)
            pos_mask    = (train_labels == 1)
            fused_train = post_arr.copy()
            fused_train[pos_mask] = pre_arr[pos_mask]   # seed positives restored to their pre-PPR score
        else:
            fused_train = score_all.reindex(train_ids_fold).to_numpy(dtype=float)
        threshold_final = find_best_threshold_by_f1(train_labels, fused_train)

        # v26: store raw (pre-remap) fused scores for test genes
        for gid, raw_s in zip(
            test_df["id"].tolist(),
            score_all.loc[test_df["id"]].to_numpy(dtype=float).tolist(),
        ):
            oof_raw_scores[gid] = raw_s

        score_all_cal   = pd.Series(
            remap_score_with_threshold(score_all.to_numpy(dtype=float), threshold_final),
            index=score_all.index, dtype=float,
        )

        test_scores  = score_all_cal.loc[test_df["id"]].to_numpy(dtype=float)
        test_metrics = evaluate_predictions(test_df["label"].to_numpy(dtype=int), test_scores)
        test_metrics["fold"]            = fold_idx
        test_metrics["n_test"]          = int(len(test_df))
        test_metrics["fusion_alpha"]    = float(best_alpha)
        test_metrics["pu_contribution"] = float(1.0 - best_alpha)
        all_metrics.append(test_metrics)

        pred_df = test_df.copy()
        pred_df["forecASD"]   = test_scores
        pred_df["pred_label"] = (pred_df["forecASD"] >= 0.5).astype(int)
        fold_dir.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(fold_dir / "test_predictions.csv", index=False)
        if fold_prior_scores is not None:
            feat_all[f"{prior_model}_empirical_prior_score"] = (
                fold_prior_scores.reindex(feat_all.index).fillna(0.5).to_numpy(dtype=float)
            )
        feat_all["score_after_prior_fusion"] = (
            score_all_pre_ppr.reindex(feat_all.index).to_numpy(dtype=float)
        )
        if ppr_scores is not None:
            feat_all["ppr_score"] = ppr_scores.reindex(feat_all.index).fillna(0.0).to_numpy(dtype=float)
        feat_all["final_raw_score_after_ppr"] = score_all.reindex(feat_all.index).to_numpy(dtype=float)
        feat_all["final_calibrated_score"] = score_all_cal.reindex(feat_all.index).to_numpy(dtype=float)
        feat_all.to_csv(fold_dir / "all_gene_component_scores.csv", index=True)

        full_scores_df = pd.DataFrame(
            {
                "gene_id": score_all_cal.index,
                "ensembl_string": score_all_cal.index,  # legacy alias for older analysis scripts
                "forecASD": score_all_cal.values,
            }
        )
        full_scores_df = full_scores_df[
            ~full_scores_df["gene_id"].isin(label_ids_set)
        ]
        full_scores_df.to_csv(fold_dir / "full_scores.csv", index=False)
        full_scores_unlabeled.append(full_scores_df.set_index("gene_id"))

        fold_info_out = {
            "fold": fold_idx,
            "model_variant": "deep_motifs_v4_lstm_prior_full_pu_rrf_no_ppr",
            "n_train": len(train_df), "n_test": len(test_df),
            "n_pos_train":  int(train_df["label"].sum()),
            "n_neg_train":  int((train_df["label"] == 0).sum()),
            "n_pos_test":   int(test_df["label"].sum()),
            "n_neg_test":   int((test_df["label"] == 0).sum()),
            "token_dim": token_dim,
            "bs_n_regions": bs_n_regions, "bs_n_timepoints": bs_n_timepoints,
            "str_token_count": str_token_count,
            "transformer_layers": transformer_layers,
            "transformer_heads": transformer_heads,
            "early_stop_metric": early_stop_metric, "early_stop_k": early_stop_k,
            "augment_factor": max(augment_factor, 1),
            "augment_scale": float(max(augment_scale, 0.0)),
            "graph_top_k": graph_top_k,
            "warmup_epochs": warmup_epochs,
            "center_update_interval": center_update_interval,
            "use_xgb_feature": prior_model == "xgb",
            "prior_model": prior_model,
            "empirical_prior_scope": "outer_fold_local" if fold_prior_scores is not None else "none",
            "fusion_mode":          fusion_mode,
            "rrf_k":                rrf_k if fusion_mode == "rrf" else None,
            "ppr_alpha":            float(ppr_alpha),
            "ppr_n_iter":           ppr_n_iter,
            "ppr_min_edge_weight":  float(ppr_min_edge_weight),
            "fusion_alpha":         float(best_alpha),
            "fusion_train_pr_auc":  float(best_fusion_auc),
            "w_sup":                float(w_sup),
            "w_pu":                 float(w_pu),
            "w_rank":               float(w_rank),
            "best_metric":          float(fit_info["best_metric"]),
            "threshold":            float(fit_info["threshold"]),
            "threshold_fused":      float(threshold_final),
            "pu_class_prior":       float(fit_info.get("pu_class_prior", np.nan)),
            "pu_class_prior_mode":  fit_info.get("pu_class_prior_mode", "auto"),
            "ebr_prior_guided_pu":  bool(fit_info.get("ebr_prior_guided_pu", False)),
            "ebr_calibration":      fit_info.get("ebr_calibration", "none"),
            "ebr_weight_floor":     float(fit_info.get("ebr_weight_floor", np.nan)),
            "ebr_uncertainty_delta": float(fit_info.get("ebr_uncertainty_delta", np.nan)),
            "ebr_rho_mean_unlabeled": float(fit_info.get("ebr_rho_mean_unlabeled", np.nan)),
            "ebr_neg_weight_mean_unlabeled": float(fit_info.get("ebr_neg_weight_mean_unlabeled", np.nan)),
            "sup_rho_mean_unlabeled": float(fit_info.get("sup_rho_mean_unlabeled", np.nan)),
            "sup_neg_weight_mean_unlabeled": float(fit_info.get("sup_neg_weight_mean_unlabeled", np.nan)),
            "n_universe_train":     int(fit_info["n_universe_train"]),
            "n_pos_train_pu":       int(fit_info["n_pos_train"]),
            "n_unlabeled_train_pu": int(fit_info["n_unlabeled_train"]),
        }
        with open(fold_dir / "fold_info.json", "w", encoding="utf-8") as f:
            json.dump(fold_info_out, f, ensure_ascii=False, indent=2)
        fold_infos.append(fold_info_out)

    # ---- v26: Global OOF threshold calibration ----
    print("[INFO] Computing global OOF threshold on full labeled set...")
    label_lookup = labels_df.set_index("id")["label"].to_dict()
    oof_ids_present = [g for g in labels_df["id"].tolist() if g in oof_raw_scores]
    oof_s_arr = np.array([oof_raw_scores[g] for g in oof_ids_present], dtype=float)
    oof_y_arr = np.array([label_lookup[g]   for g in oof_ids_present], dtype=int)
    # beta=0.8: precision-biased F-beta prevents the pooled OOF distribution
    # from driving the threshold too low (which collapses precision vs per-fold).
    global_threshold = find_best_threshold_by_f1(oof_y_arr, oof_s_arr, beta=0.8)
    print(f"[INFO] Global OOF threshold (F-beta=0.8): {global_threshold:.4f}")

    global_metrics_list = []
    for fold_idx_g, (_, test_idx_g) in enumerate(skf.split(X_ids, y), start=1):
        test_df_g  = labels_df.iloc[test_idx_g]
        test_ids_g = test_df_g["id"].tolist()
        test_y_g   = test_df_g["label"].to_numpy(dtype=int)
        test_raw_g = np.array([oof_raw_scores.get(g, 0.5) for g in test_ids_g], dtype=float)
        test_cal_g = remap_score_with_threshold(test_raw_g, global_threshold)
        m_g = evaluate_predictions(test_y_g, test_cal_g)
        m_g["fold"]           = fold_idx_g
        m_g["n_test"]         = len(test_df_g)
        m_g["fusion_alpha"]   = float("nan")
        m_g["pu_contribution"] = float("nan")
        global_metrics_list.append(m_g)

    global_metrics_df = pd.DataFrame(global_metrics_list)
    _global_metric_cols = [c for c in [
        "fold", "n_test",
        "accuracy", "precision", "recall", "f1",
        "macro_f1", "weighted_f1", "pr_auc", "roc_auc",
        "precision@10", "recall@10", "lift@10", "ndcg@10",
        "precision@20", "recall@20", "lift@20", "ndcg@20",
        "precision@50", "recall@50", "lift@50", "ndcg@50",
        "fusion_alpha", "pu_contribution",
    ] if c in global_metrics_df.columns]
    global_metrics_df = global_metrics_df[_global_metric_cols]
    global_metrics_df.to_csv(output_dir / "cv_fold_metrics_global_threshold.csv", index=False)

    global_summary_rows = []
    for col in [c for c in _global_metric_cols if c != "fold"]:
        mean, std = nanmean_std(global_metrics_df[col])
        global_summary_rows.append({
            "metric": col,
            "mean": mean,
            "std":  std,
        })
    pd.DataFrame(global_summary_rows).to_csv(
        output_dir / "cv_metrics_summary_global_threshold.csv", index=False)
    print(f"[INFO] Global threshold={global_threshold:.4f} → cv_metrics_summary_global_threshold.csv")

    metrics_df = pd.DataFrame(all_metrics)
    metric_cols = [
        "fold", "n_test",
        "accuracy", "precision", "recall", "f1",
        "macro_f1", "weighted_f1", "pr_auc", "roc_auc",
        "precision@10", "recall@10", "lift@10", "ndcg@10",
        "precision@20", "recall@20", "lift@20", "ndcg@20",
        "precision@50", "recall@50", "lift@50", "ndcg@50",
        "fusion_alpha", "pu_contribution",
    ]
    metrics_df = metrics_df[metric_cols]
    metrics_df.to_csv(output_dir / "cv_fold_metrics.csv", index=False)

    summary_rows = []
    for col in [c for c in metric_cols if c != "fold"]:
        mean, std = nanmean_std(metrics_df[col])
        summary_rows.append({
            "metric": col,
            "mean":   mean,
            "std":    std,
        })
    pd.DataFrame(summary_rows).to_csv(output_dir / "cv_metrics_summary.csv", index=False)

    if full_scores_unlabeled:
        (pd.concat(full_scores_unlabeled).groupby(level=0).mean(numeric_only=True)
         .reset_index().rename(columns={"index": "gene_id"})
         .assign(ensembl_string=lambda df: df["gene_id"])
         .to_csv(output_dir / "full_scores_summary.csv", index=False))
    if fold_infos:
        pd.DataFrame(fold_infos).to_csv(output_dir / "fold_infos_summary.csv", index=False)

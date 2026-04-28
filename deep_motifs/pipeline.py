from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from .xgb import (
    build_fold_string_feature_matrix,
    compute_graph_features,
    evaluate_predictions,
)

from .features import compute_xgb_oof_scores, poly_expand_meta
from .fusion import asymmetric_rrf_fuse, fuse_scores, rrf_fuse_scores, search_optimal_alpha
from .graph import build_weighted_string_graph
from .metrics import find_best_threshold_by_f1, recall_at_k_score, remap_score_with_threshold
from .models import DeepMOTIFs
from .ppr import compute_ppr_from_seeds, propagate_scores_ppr
from .pretrain import compute_pretrain_meta_importance, pretrain_encoder
from .training import fit_deep_motifs_and_export
from .utils import (
    _apply_feature_noise,
    _apply_label_noise,
    build_view_frames,
    standardize_fit_and_all,
)


# ============================================================
# Cross-validation driver
# ============================================================

def run_pu(
    labels_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    brainspan_df: pd.DataFrame,
    G,
    data_dir: Path,
    output_dir: Path,
    string_mode: str,
    max_string_anchors: int,
    n_splits: int,
    random_state: int,
    force_rebuild_graph_features: bool,
    device,
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
    w_bce: float,
    w_pu: float,
    w_rank: float,
    w_oc: float,
    w_graph: float,
    w_cons: float,
    graph_top_k: int,
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
    ckpt_avg_k: int = 5,
    fusion_mode: str = "fixed",
    rrf_k: int = 60,
    ppr_alpha: float = 1.0,
    ppr_n_iter: int = 30,
    ppr_min_edge_weight: float = 0.5,
    ppr_fusion_weight: float = 0.7,
    poly_top_k: int = 6,          # v25: top-K meta features for polynomial expansion
    ablate_string: bool = False,
    ablate_brainspan: bool = False,
    noise_type: str = "none",
    noise_level: float = 0.0,
    label_flip_rate: float = 0.0,
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

    # 鏋勫缓甯︽潈閲嶇殑 STRING 鍥撅紝鐢ㄤ簬鍔犳潈 GCN 棰勮仛鍚?    print("[INFO] Building weighted STRING graph for GCN aggregation...")
    weighted_G = build_weighted_string_graph(
        data_dir=data_dir,
        score_threshold=400,
        cache_path=output_dir.parent / "cache" / "string_graph_weighted.pkl",
    )
    print(
        f"[INFO] Weighted STRING graph: "
        f"{weighted_G.number_of_nodes()} nodes, {weighted_G.number_of_edges()} edges"
    )

    # STRING graph features (graph mode)
    string_graph_features: pd.DataFrame | None = None
    if string_mode == "graph":
        cache_path = output_dir.parent / "cache" / "string_graph_features.pkl"
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

    # Pre-compute XGBoost OOF scores
    xgb_oof_scores: pd.Series | None = None
    if use_xgb_feature:
        cache_dir = output_dir.parent / "cache"
        label_count = int(len(labels_df))
        pos_count = int(labels_df["label"].sum())
        xgb_extra_tag = ""
        if "xgb_max_depth" in locals():
            reg_alpha_tag = str(xgb_reg_alpha).replace(".", "p")
            gamma_tag = str(xgb_gamma).replace(".", "p")
            xgb_extra_tag = (
                f"_d{xgb_max_depth}_mcw{xgb_min_child_weight}"
                f"_ra{reg_alpha_tag}_g{gamma_tag}"
            )
        xgb_cache_tag = (
            f"{Path(__file__).stem}_{Path(output_dir).name}_"
            f"s{n_splits}_rs{random_state}_"
            f"n{label_count}_p{pos_count}_est{xgb_n_estimators}{xgb_extra_tag}"
        )
        xgb_cache_path = cache_dir / f"xgb_oof_scores_{xgb_cache_tag}.csv"
        if force_rebuild_xgb_oof and xgb_cache_path.exists():
            xgb_cache_path.unlink()
            print("[INFO] XGBoost OOF cache deleted 鈥?will recompute.")

        print("[INFO] Pre-computing XGBoost OOF scores (no leakage)...")
        oof_string_df = build_fold_string_feature_matrix(
            G=G, target_ids=target_ids,
            anchor_ids=labels_df["id"].tolist(),
            max_anchors=max_string_anchors,
        )
        oof_meta, oof_bs, oof_str = build_view_frames(
            meta_df=meta_df, brainspan_df=brainspan_df, string_df=oof_string_df,
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
        xgb_oof_scores = compute_xgb_oof_scores(
            labels_df=labels_df,
            meta_all=oof_meta, bs_all=oof_bs, str_all=oof_str,
            n_splits=n_splits, random_state=random_state,
            n_estimators=xgb_n_estimators,
            xgb_max_depth=xgb_max_depth,
            xgb_min_child_weight=xgb_min_child_weight,
            xgb_reg_alpha=xgb_reg_alpha,
            xgb_gamma=xgb_gamma,
            cache_path=xgb_cache_path,
        )
        pos_ids_set = set(labels_df[labels_df["label"] == 1]["id"])
        neg_ids_set = set(labels_df[labels_df["label"] == 0]["id"])
        print(
            f"[INFO] XGBoost OOF done. "
            f"pos_mean={xgb_oof_scores.loc[list(pos_ids_set)].mean():.4f}  "
            f"neg_mean={xgb_oof_scores.loc[list(neg_ids_set)].mean():.4f}"
        )
        xgb_oof_scores.to_csv(output_dir / "xgb_oof_scores.csv", header=["xgb_oof_score"])

    # v25: Pretrain reconstruction-error polynomial meta-feature expansion.
    # A temporary Deep-MOTIFs is pretrained (fully unsupervised) on ALL genes using
    # masked-feature reconstruction.  Per-feature ablation MSE ranks features with
    # ZERO label leakage — no labels are touched during importance computation.
    # The same top-K pairs/squares are used for polynomial expansion in every fold.
    top_meta_pairs:   list = []
    top_meta_squares: list = []
    if use_xgb_feature and poly_top_k > 0:
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
        _meta_t_pt  = import_torch().from_numpy(_x_meta_pt).float()
        _bs_t_pt    = import_torch().from_numpy(_x_bs_pt).float()

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

        _pos_ids_set_pt = set(labels_df[labels_df["label"] == 1]["id"])
        _pos_global_idx_pt = np.asarray(
            [i for i, gid in enumerate(target_ids) if gid in _pos_ids_set_pt],
            dtype=np.int64,
        )

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
            w_pretrain_pu=w_pretrain_pu,
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
            w_bce=w_bce,
            w_pu=w_pu,
            w_rank=w_rank,
            w_oc=w_oc,
            w_graph=w_graph,
            w_cons=w_cons,
            graph_top_k=graph_top_k,
            xgb_oof_scores=xgb_oof_scores,
            weighted_G=weighted_G,
            gcn_n_layers=gcn_n_layers,
            warmup_epochs=warmup_epochs,
            center_update_interval=center_update_interval,
            use_torch_compile=use_torch_compile,
            pretrain_epochs=pretrain_epochs,
            pretrain_lr=pretrain_lr,
            pretrain_mask_rate=pretrain_mask_rate,
            w_pretrain_pu=w_pretrain_pu,
            ckpt_avg_k=ckpt_avg_k,
            progress_prefix=f"[Fold {fold_idx}][PU] ",
        )

        # v4: 鍥哄畾 alpha=0.5锛屼笉鍐嶅湪璁粌闆嗕笂鎼滅储銆?        # v8: 鏂板 fusion_mode="rrf" 閫夐」锛岀敤 Reciprocal Rank Fusion 浠ｆ浛绾挎€у姞鏉冦€?        best_alpha = 0.5
        best_fusion_auc = float("nan")
        if xgb_oof_scores is not None:
            xgb_all = xgb_oof_scores.reindex(score_all.index).fillna(0.5).to_numpy(dtype=float)
            pu_all  = score_all.to_numpy(dtype=float)

            if fusion_mode == "rrf":
                print(
                    f"[Fold {fold_idx}] 铻嶅悎: RRF (k={rrf_k})"
                )
                fused_all = rrf_fuse_scores(xgb_all, pu_all, k=rrf_k)
                best_alpha = -1.0   # sentinel: indicates RRF mode in CSV
            else:
                print(
                    f"[Fold {fold_idx}] 铻嶅悎: fixed alpha=0.50 (XGB鏉冮噸=50%, PU鏉冮噸=50%)"
                )
                fused_all = fuse_scores(xgb_all, pu_all, alpha=best_alpha)

            score_all = pd.Series(fused_all, index=score_all.index, dtype=float)

        # v17: Seeded PPR 鈥?propagate FROM known training positives through STRING
        # v18: save pre-PPR scores for threshold calibration
        # v20: confidence-weighted seeds (XGBoost OOF) + asymmetric PPR fusion weight
        score_all_pre_ppr = score_all.copy()
        if ppr_alpha < 1.0:
            train_pos_ids = train_df[train_df["label"] == 1]["id"].tolist()

            # v20: weight each seed by its XGBoost OOF confidence score
            # High-confidence positives contribute more to PPR 鈫?less noise propagation
            if use_xgb_feature and xgb_oof_scores is not None:
                raw_w   = {gid: float(xgb_oof_scores.get(gid, 0.5)) for gid in train_pos_ids}
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

        # v19: 娣峰悎鏍″噯 鈥?闃虫€х敤 pre-PPR锛堥伩鍏嶇瀛愯啫鑳€锛夛紝闃存€х敤 post-PPR锛堜笌娴嬭瘯闆嗗悓鍒嗗竷锛?        # 璁粌闃虫€ф槸 PPR 绉嶅瓙锛宲ost-PPR 鍒嗘暟鈮?.0锛堜汉宸ヨ啫鑳€锛夆啋 杩樺師涓?pre-PPR 鍒嗘暟
        # use train-set scores to calibrate threshold consistently with test distribution
        train_ids_fold  = fit_info["train_ids"]
        train_labels    = np.asarray(fit_info["train_labels"], dtype=int)
        if ppr_alpha < 1.0:
            post_arr    = score_all.reindex(train_ids_fold).to_numpy(dtype=float)
            pre_arr     = score_all_pre_ppr.reindex(train_ids_fold).to_numpy(dtype=float)
            pos_mask    = (train_labels == 1)
            fused_train = post_arr.copy()
            fused_train[pos_mask] = pre_arr[pos_mask]   # 绉嶅瓙闃虫€ц繕鍘熶负 pre-PPR 鍒嗘暟
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
        pred_df.to_csv(fold_dir / "test_predictions.csv", index=False)
        feat_all.to_csv(fold_dir / "all_gene_component_scores.csv", index=True)

        full_scores_df = pd.DataFrame(
            {"ensembl_string": score_all_cal.index, "forecASD": score_all_cal.values}
        )
        full_scores_df = full_scores_df[
            ~full_scores_df["ensembl_string"].isin(label_ids_set)
        ]
        full_scores_df.to_csv(fold_dir / "full_scores.csv", index=False)
        full_scores_unlabeled.append(full_scores_df.set_index("ensembl_string"))

        fold_info_out = {
            "fold": fold_idx,
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
            "use_xgb_feature": use_xgb_feature,
            "fusion_mode":          fusion_mode,
            "rrf_k":                rrf_k if fusion_mode == "rrf" else None,
            "ppr_alpha":            float(ppr_alpha),
            "ppr_n_iter":           ppr_n_iter,
            "ppr_min_edge_weight":  float(ppr_min_edge_weight),
            "fusion_alpha":         float(best_alpha),
            "fusion_train_pr_auc":  float(best_fusion_auc),
            "best_metric":          float(fit_info["best_metric"]),
            "threshold":            float(fit_info["threshold"]),
            "threshold_fused":      float(threshold_final),
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
        vals = pd.to_numeric(global_metrics_df[col], errors="coerce")
        global_summary_rows.append({
            "metric": col,
            "mean": float(np.nanmean(vals)),
            "std":  float(np.nanstd(vals, ddof=1)) if vals.notna().sum() > 1 else 0.0,
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
        vals = pd.to_numeric(metrics_df[col], errors="coerce")
        summary_rows.append({
            "metric": col,
            "mean":   float(np.nanmean(vals)),
            "std":    float(np.nanstd(vals, ddof=1)) if vals.notna().sum() > 1 else 0.0,
        })
    pd.DataFrame(summary_rows).to_csv(output_dir / "cv_metrics_summary.csv", index=False)

    if full_scores_unlabeled:
        (pd.concat(full_scores_unlabeled).groupby(level=0).mean(numeric_only=True)
         .reset_index().rename(columns={"index": "ensembl_string"})
         .to_csv(output_dir / "full_scores_summary.csv", index=False))
    if fold_infos:
        pd.DataFrame(fold_infos).to_csv(output_dir / "fold_infos_summary.csv", index=False)


def import_torch():
    import torch
    return torch
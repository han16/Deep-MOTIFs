from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# The LSTM empirical-prior estimator (``--prior-model lstm``) depends on
# ``train_lstm``/``score_lstm`` from a standalone ``lstm.py`` that is not shipped
# in this workspace. Import it if available, otherwise defer the failure: the
# package still imports and runs for ``--prior-model xgb`` / ``none``, and a clear
# error is raised only if the LSTM prior is actually requested.
try:
    from lstm import score_lstm
    from lstm import train_lstm
except ModuleNotFoundError:
    try:
        from nouse.lstm import score_lstm
        from nouse.lstm import train_lstm
    except ModuleNotFoundError:
        def _lstm_unavailable(*_args, **_kwargs):
            raise ModuleNotFoundError(
                "lstm.py (providing train_lstm/score_lstm) is not present in this "
                "workspace, so the LSTM empirical prior is unavailable. Run with "
                "--prior-model xgb (or --prior-model none) instead, or add lstm.py."
            )

        score_lstm = _lstm_unavailable
        train_lstm = _lstm_unavailable


def _build_xgb_feature_matrix(
    meta_all: pd.DataFrame,
    bs_all: pd.DataFrame,
    str_all: pd.DataFrame,
) -> pd.DataFrame:
    X = pd.concat(
        [meta_all.astype(np.float32),
         bs_all.astype(np.float32),
         str_all.astype(np.float32)],
        axis=1,
    )
    X.columns = [str(c) for c in X.columns]
    return X.loc[:, ~X.columns.duplicated()]


def _fit_xgb_v18(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_all: pd.DataFrame,
    n_estimators: int,
    random_state: int,
    max_depth: int = 4,
    min_child_weight: int = 5,
    reg_alpha: float = 0.1,
    gamma: float = 0.1,
) -> tuple:
    """
    v18 optimised XGBoost fit.  Compared to the default fit_xgb_and_score:
      max_depth        6  鈫?4    (shallower trees, less overfitting on ~665 labels)
      min_child_weight 1  鈫?5    (require more samples per leaf)
      reg_alpha        0  鈫?0.1  (L1 regularisation for feature sparsity)
      gamma            0  鈫?0.1  (minimum gain required to make a split)
    """
    from xgboost import XGBClassifier
    from sklearn.dummy import DummyClassifier

    y_unique = np.unique(y_train)
    if y_unique.size < 2:
        constant = int(y_unique[0]) if y_unique.size == 1 else 0
        clf = DummyClassifier(strategy="constant", constant=constant)
        clf.fit(X_train, y_train)
        scores = np.full(X_all.shape[0], float(constant), dtype=float)
        return clf, pd.Series(scores, index=X_all.index)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        learning_rate=0.05,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        reg_lambda=1.0,
        reg_alpha=reg_alpha,
        gamma=gamma,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_all)
    pos_col = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
    return clf, pd.Series(proba[:, pos_col], index=X_all.index, dtype=float)


def compute_xgb_oof_scores(
    labels_df: pd.DataFrame,
    meta_all: pd.DataFrame,
    bs_all: pd.DataFrame,
    str_all: pd.DataFrame,
    n_splits: int,
    random_state: int,
    n_estimators: int = 500,
    xgb_max_depth: int = 4,
    xgb_min_child_weight: int = 5,
    xgb_reg_alpha: float = 0.1,
    xgb_gamma: float = 0.1,
    cache_path: Path | None = None,
) -> pd.Series:
    """Out-of-fold XGBoost scores 鈥?zero leakage. v18: optimised hyperparameters."""
    if cache_path is not None and cache_path.exists():
        print(f"[INFO] Loading XGBoost OOF scores from cache: {cache_path}")
        return pd.read_csv(cache_path, index_col=0).iloc[:, 0]

    X_all = _build_xgb_feature_matrix(meta_all, bs_all, str_all)
    label_ids = labels_df["id"].tolist()
    y_all     = labels_df["label"].to_numpy(dtype=int)
    oof_scores = pd.Series(0.5, index=X_all.index, dtype=float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + 999)
    for fold_i, (inner_tr, inner_val) in enumerate(skf.split(label_ids, y_all), start=1):
        tr_ids  = [label_ids[i] for i in inner_tr]
        val_ids = [label_ids[i] for i in inner_val]
        _, fold_scores = _fit_xgb_v18(
            X_train=X_all.loc[tr_ids],
            y_train=y_all[inner_tr],
            X_all=X_all.loc[val_ids],
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=xgb_max_depth,
            min_child_weight=xgb_min_child_weight,
            reg_alpha=xgb_reg_alpha,
            gamma=xgb_gamma,
        )
        oof_scores.loc[val_ids] = fold_scores.values
        print(f"  [XGB-OOF] inner fold {fold_i}/{n_splits} done")

    label_id_set   = set(label_ids)
    unlabelled_ids = [i for i in X_all.index if i not in label_id_set]
    if unlabelled_ids:
        _, full_scores = _fit_xgb_v18(
            X_train=X_all.loc[label_ids],
            y_train=y_all,
            X_all=X_all.loc[unlabelled_ids],
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=xgb_max_depth,
            min_child_weight=xgb_min_child_weight,
            reg_alpha=xgb_reg_alpha,
            gamma=xgb_gamma,
        )
        oof_scores.loc[unlabelled_ids] = full_scores.values

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        oof_scores.to_csv(cache_path, header=["xgb_oof_score"])
        print(f"[INFO] XGBoost OOF scores cached to: {cache_path}")

    return oof_scores


def compute_xgb_fold_local_prior_scores(
    train_df: pd.DataFrame,
    meta_all: pd.DataFrame,
    bs_all: pd.DataFrame,
    str_all: pd.DataFrame,
    n_splits: int,
    random_state: int,
    n_estimators: int = 500,
    xgb_max_depth: int = 4,
    xgb_min_child_weight: int = 5,
    xgb_reg_alpha: float = 0.1,
    xgb_gamma: float = 0.1,
) -> pd.Series:
    """
    Fold-local empirical prior estimator.

    For the outer training labels, scores are out-of-fold predictions from an
    inner CV. For every gene outside the outer training labels, scores are from
    one XGBoost model fitted only on the outer training labels. This keeps the
    empirical prior aligned with the outer CV protocol: no outer-test labels are
    used to construct the prior estimator for that fold.
    """
    X_all = _build_xgb_feature_matrix(meta_all, bs_all, str_all)
    train_ids = train_df["id"].astype(str).tolist()
    y_train = train_df["label"].to_numpy(dtype=int)

    scores = pd.Series(0.5, index=X_all.index, dtype=float)
    class_counts = np.bincount(y_train, minlength=2)
    inner_splits = min(int(n_splits), int(class_counts.min()))

    if inner_splits >= 2 and np.unique(y_train).size == 2:
        skf_inner = StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=random_state + 1999
        )
        for inner_i, (inner_tr, inner_val) in enumerate(
            skf_inner.split(train_ids, y_train), start=1
        ):
            tr_ids = [train_ids[i] for i in inner_tr]
            val_ids = [train_ids[i] for i in inner_val]
            _, val_scores = _fit_xgb_v18(
                X_train=X_all.loc[tr_ids],
                y_train=y_train[inner_tr],
                X_all=X_all.loc[val_ids],
                n_estimators=n_estimators,
                random_state=random_state + inner_i,
                max_depth=xgb_max_depth,
                min_child_weight=xgb_min_child_weight,
                reg_alpha=xgb_reg_alpha,
                gamma=xgb_gamma,
            )
            scores.loc[val_ids] = val_scores.values
            print(f"  [XGB-prior fold-local] inner fold {inner_i}/{inner_splits} done")

    outside_train_ids = [gid for gid in X_all.index if gid not in set(train_ids)]
    if outside_train_ids:
        _, outside_scores = _fit_xgb_v18(
            X_train=X_all.loc[train_ids],
            y_train=y_train,
            X_all=X_all.loc[outside_train_ids],
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=xgb_max_depth,
            min_child_weight=xgb_min_child_weight,
            reg_alpha=xgb_reg_alpha,
            gamma=xgb_gamma,
        )
        scores.loc[outside_train_ids] = outside_scores.values

    return scores.astype(float)


def _fit_lstm_prior(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_score: pd.DataFrame,
    meta_dim: int,
    bs_dim: int,
    str_dim: int,
    bs_n_regions: int,
    bs_n_timepoints: int,
    device: torch.device,
    random_state: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> pd.Series:
    y_unique = np.unique(y_train)
    if y_unique.size < 2:
        constant = float(y_unique[0]) if y_unique.size == 1 else 0.5
        return pd.Series(constant, index=X_score.index, dtype=float)

    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train.to_numpy(dtype=np.float32))
    X_score_np = scaler.transform(X_score.to_numpy(dtype=np.float32))

    model = train_lstm(
        X_train_np=X_train_np,
        y_train_np=y_train.astype(int),
        meta_dim=meta_dim,
        bs_dim=bs_dim,
        str_dim=str_dim,
        bs_n_regions=bs_n_regions,
        bs_n_timepoints=bs_n_timepoints,
        device=device,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        random_state=random_state,
    )
    scores = score_lstm(model, X_score_np, device=device, batch_size=batch_size)
    return pd.Series(scores, index=X_score.index, dtype=float)


def compute_lstm_fold_local_prior_scores(
    train_df: pd.DataFrame,
    meta_all: pd.DataFrame,
    bs_all: pd.DataFrame,
    str_all: pd.DataFrame,
    n_splits: int,
    random_state: int,
    device: torch.device,
    bs_n_regions: int,
    bs_n_timepoints: int,
    epochs: int = 120,
    batch_size: int = 128,
    lr: float = 5e-4,
    patience: int = 20,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> pd.Series:
    """
    Fold-local LSTM empirical prior estimator.

    Outer-train labels receive inner-CV out-of-fold predictions; all genes
    outside the outer train labels are scored by a model fitted only on the
    outer train labels. This mirrors the v3 XGBoost prior protocol without
    using outer-test labels.
    """
    X_all = _build_xgb_feature_matrix(meta_all, bs_all, str_all)
    train_ids = train_df["id"].astype(str).tolist()
    y_train = train_df["label"].to_numpy(dtype=int)
    meta_dim = int(meta_all.shape[1])
    bs_dim = int(bs_all.shape[1])
    str_dim = int(str_all.shape[1])
    if bs_dim > 0 and bs_dim != bs_n_regions * bs_n_timepoints:
        bs_n_timepoints = min(bs_n_timepoints, bs_dim)
        bs_n_regions = max(bs_dim // max(bs_n_timepoints, 1), 1)

    scores = pd.Series(0.5, index=X_all.index, dtype=float)
    class_counts = np.bincount(y_train, minlength=2)
    inner_splits = min(int(n_splits), int(class_counts.min()))

    if inner_splits >= 2 and np.unique(y_train).size == 2:
        skf_inner = StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=random_state + 2999
        )
        for inner_i, (inner_tr, inner_val) in enumerate(
            skf_inner.split(train_ids, y_train), start=1
        ):
            tr_ids = [train_ids[i] for i in inner_tr]
            val_ids = [train_ids[i] for i in inner_val]
            val_scores = _fit_lstm_prior(
                X_train=X_all.loc[tr_ids],
                y_train=y_train[inner_tr],
                X_score=X_all.loc[val_ids],
                meta_dim=meta_dim,
                bs_dim=bs_dim,
                str_dim=str_dim,
                bs_n_regions=bs_n_regions,
                bs_n_timepoints=bs_n_timepoints,
                device=device,
                random_state=random_state + inner_i,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            scores.loc[val_ids] = val_scores.values
            print(f"  [LSTM-prior fold-local] inner fold {inner_i}/{inner_splits} done")

    outside_train_ids = [gid for gid in X_all.index if gid not in set(train_ids)]
    if outside_train_ids:
        outside_scores = _fit_lstm_prior(
            X_train=X_all.loc[train_ids],
            y_train=y_train,
            X_score=X_all.loc[outside_train_ids],
            meta_dim=meta_dim,
            bs_dim=bs_dim,
            str_dim=str_dim,
            bs_n_regions=bs_n_regions,
            bs_n_timepoints=bs_n_timepoints,
            device=device,
            random_state=random_state,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        scores.loc[outside_train_ids] = outside_scores.values

    return scores.astype(float)

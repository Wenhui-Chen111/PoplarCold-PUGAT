# -*- coding: utf-8 -*-
"""XGBoost baseline for PoplarCold-PUGAT study.

The model uses X.npy only. Graphs are intentionally unused because XGBoost is the
feature-only comparator named in the manuscript. Training is a PN surrogate:
training positives are class 1; true-unlabeled genes (y.npy == 0) are treated as
class 0 with scale_pos_weight. Validation/test positives are never used for fitting.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np

from baseline_common import (
    BaselineData, SplitMasks, base_parser, compute_ranking_metrics,
    parameter_grid, prepare_from_args, run_repeated_baseline, validation_key,
)

METHOD_NAME = "XGBoost"
METHOD_SPEC: Dict[str, Any] = {
    "method": METHOD_NAME,
    "role": "feature-only tabular baseline",
    "node_features": "Exact X.npy used by the main model; no extra biological features.",
    "network_information": "None by method design.",
    "training_labels": (
        "Train positives are positive; y.npy==0 true-unlabeled genes are presumed negative. "
        "Validation/test positives are excluded from the fitting pool."
    ),
    "loss": "XGBoost binary logistic objective with scale_pos_weight = n_unlabeled/n_train_positive.",
    "sampling": "No negative subsampling; all true-unlabeled genes are used in every fit.",
    "tuning": (
        "Grid search on the fixed validation-positive + true-unlabeled universe. "
        "Selection metric = 0.6*nDCG@100 + 0.4*Recall@100."
    ),
    "training_budget": "Default 12 configurations, 500 boosting rounds per configuration; no silent fallback model.",
    "transductive_note": "All node features are available, but held-out positive labels are not used for fitting.",
}


def fit_score(
    data: BaselineData,
    masks: SplitMasks,
    seed: int,
    *,
    max_trials: int = 12,
    n_estimators: int = 500,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "XGBoost baseline requires the 'xgboost' package. "
            "No GradientBoosting fallback is used because that would change the baseline method."
        ) from exc

    train_mask = masks.train_pos | masks.true_unlabeled
    y_train = masks.train_pos[train_mask].astype(np.int64)
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("XGBoost requires at least one training positive and one true-unlabeled sample")
    scale_pos_weight = float(n_neg / n_pos)

    grid = parameter_grid(
        {
            "max_depth": [3, 6, 9],
            "learning_rate": [0.03, 0.1],
            "min_child_weight": [1.0, 5.0],
        },
        max_trials=max_trials,
    )

    best_key = -np.inf
    best_scores = None
    best_cfg: Dict[str, Any] = {}
    trial_records = []

    for trial_id, cfg in enumerate(grid, start=1):
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=n_estimators,
            max_depth=int(cfg["max_depth"]),
            learning_rate=float(cfg["learning_rate"]),
            min_child_weight=float(cfg["min_child_weight"]),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=0,
        )
        model.fit(data.X[train_mask], y_train)
        scores = model.predict_proba(data.X)[:, 1].astype(float)
        val_metrics = compute_ranking_metrics(scores, masks.val_pos, masks.true_unlabeled)
        key = validation_key(val_metrics)
        trial_records.append({"trial": trial_id, "val_key": key, **cfg})
        if key > best_key:
            best_key = key
            best_scores = scores.copy()
            best_cfg = dict(cfg)

    if best_scores is None:
        raise RuntimeError("XGBoost search produced no model")

    best_cfg.update({
        "n_estimators": int(n_estimators),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
    })
    audit = {
        "n_tuning_trials": len(grid),
        "validation_selection_key": "0.6*nDCG@100 + 0.4*Recall@100",
        "training_samples": int(train_mask.sum()),
        "training_positives": n_pos,
        "presumed_negatives": n_neg,
        "trees_per_trial": int(n_estimators),
        "best_validation_key": float(best_key),
        "trial_records": trial_records,
    }
    return best_scores, best_cfg, audit


def main() -> None:
    parser = base_parser("Reviewer-facing XGBoost baseline")
    parser.add_argument("--max_trials", type=int, default=12)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    data, manifest = prepare_from_args(args)
    run_repeated_baseline(
        data=data,
        manifest=manifest,
        out_dir=args.out_dir,
        method_name=METHOD_NAME,
        method_spec=METHOD_SPEC,
        fit_score_fn=lambda d, m, s: fit_score(
            d, m, s,
            max_trials=args.max_trials,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
        ),
        train_seed_base=args.train_seed_base,
        save_scores=not args.no_save_scores,
    )


if __name__ == "__main__":
    main()

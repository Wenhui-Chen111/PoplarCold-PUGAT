# -*- coding: utf-8 -*-
"""PU-Bagging baseline using the Mordelet & Vert bagging-SVM scheme.

Each bag trains an SVM on all training positives plus a bootstrap sample from the
true-unlabeled pool, temporarily treating sampled unlabeled genes as negatives.
Transductive scores are averaged over out-of-bag predictions. Validation/test seed
positives are never sampled as negatives and are never used as labels during fitting.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from baseline_common import (
    BaselineData, SplitMasks, base_parser, compute_ranking_metrics,
    parameter_grid, prepare_from_args, run_repeated_baseline, validation_key,
)

METHOD_NAME = "PU-Bagging"
METHOD_SPEC: Dict[str, Any] = {
    "method": METHOD_NAME,
    "role": "classical positive-unlabeled baseline",
    "reference_implementation_concept": "Mordelet & Vert bagging SVM: positives vs bootstrap subsamples of U; average predictions.",
    "node_features": "Exact X.npy source features; z-scored by a label-independent scaler fitted on all graph nodes.",
    "network_information": "None by method design.",
    "training_labels": "Only manifest training positives are labeled positive; y.npy==0 genes form U.",
    "loss": "SVM hinge-loss classifier inside each bag (scikit-learn SVC).",
    "sampling": (
        "Every bag uses all training positives and a bootstrap sample with replacement from true U. "
        "Transductive U scores use out-of-bag averaging; held-out positives are always out-of-bag because they are not in U."
    ),
    "tuning": "C, gamma and U:positive bootstrap ratio selected on the fixed validation universe.",
    "training_budget": "Default 12 hyperparameter configurations x 100 SVM bags each.",
    "transductive_note": "This is the transductive PU-Bagging setting appropriate for gene ranking.",
}

GammaType = Union[str, float]


def _bagging_svm_scores(
    X: np.ndarray,
    masks: SplitMasks,
    seed: int,
    *,
    C: float,
    gamma: GammaType,
    unlabeled_ratio: float,
    n_bags: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    pos_idx = np.where(masks.train_pos)[0]
    un_idx = np.where(masks.true_unlabeled)[0]
    n_pos = len(pos_idx)
    if n_pos == 0 or len(un_idx) == 0:
        raise ValueError("PU-Bagging requires training positives and true-unlabeled genes")

    bag_size = min(len(un_idx), max(1, int(round(unlabeled_ratio * n_pos))))
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    oob_sum = np.zeros(n, dtype=np.float64)
    oob_count = np.zeros(n, dtype=np.int64)
    all_sum = np.zeros(n, dtype=np.float64)

    heldout_mask = masks.val_pos | masks.test_pos
    candidate_mask = masks.true_unlabeled | heldout_mask
    candidate_idx = np.where(candidate_mask)[0]

    for _ in range(n_bags):
        sampled = rng.choice(un_idx, size=bag_size, replace=True)
        train_idx = np.concatenate([pos_idx, sampled])
        y_train = np.concatenate([
            np.ones(len(pos_idx), dtype=np.int64),
            np.zeros(len(sampled), dtype=np.int64),
        ])

        clf = SVC(
            C=float(C),
            kernel="rbf",
            gamma=gamma,
            class_weight="balanced",
            probability=False,
            random_state=int(rng.randint(0, 2**31 - 1)),
        )
        clf.fit(X[train_idx], y_train)
        full_scores = clf.decision_function(X).astype(np.float64)
        all_sum += full_scores

        sampled_unique = np.unique(sampled)
        in_bag = np.zeros(n, dtype=bool)
        in_bag[sampled_unique] = True
        oob_idx = candidate_idx[~in_bag[candidate_idx]]
        oob_sum[oob_idx] += full_scores[oob_idx]
        oob_count[oob_idx] += 1

    scores = all_sum / float(n_bags)
    have_oob = oob_count > 0
    scores[have_oob] = oob_sum[have_oob] / oob_count[have_oob]

    audit = {
        "n_bags": int(n_bags),
        "unlabeled_bootstrap_size": int(bag_size),
        "n_train_positive": int(n_pos),
        "n_true_unlabeled": int(len(un_idx)),
        "candidate_nodes_with_no_oob_prediction": int((candidate_mask & ~have_oob).sum()),
    }
    return scores, audit


def fit_score(
    data: BaselineData,
    masks: SplitMasks,
    seed: int,
    *,
    max_trials: int = 12,
    n_bags: int = 100,
) -> Tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
    # Label-independent transform: avoids using held-out label status in preprocessing.
    X_scaled = StandardScaler().fit_transform(data.X).astype(np.float32)

    grid = parameter_grid(
        {
            "C": [1.0, 10.0, 100.0],
            "gamma": ["scale", 0.1],
            "unlabeled_ratio": [1.0, 2.0],
        },
        max_trials=max_trials,
    )

    best_key = -np.inf
    best_scores = None
    best_cfg: Dict[str, Any] = {}
    best_inner_audit: Dict[str, Any] = {}
    trial_records = []

    for trial_id, cfg in enumerate(grid, start=1):
        scores, inner = _bagging_svm_scores(
            X_scaled,
            masks,
            seed,
            C=float(cfg["C"]),
            gamma=cfg["gamma"],
            unlabeled_ratio=float(cfg["unlabeled_ratio"]),
            n_bags=n_bags,
        )
        val_metrics = compute_ranking_metrics(scores, masks.val_pos, masks.true_unlabeled)
        key = validation_key(val_metrics)
        trial_records.append({"trial": trial_id, "val_key": key, **cfg})
        if key > best_key:
            best_key = key
            best_scores = scores.copy()
            best_cfg = dict(cfg)
            best_inner_audit = inner

    if best_scores is None:
        raise RuntimeError("PU-Bagging search produced no model")

    best_cfg["n_bags"] = int(n_bags)
    audit = {
        "n_tuning_trials": len(grid),
        "validation_selection_key": "0.6*nDCG@100 + 0.4*Recall@100",
        "best_validation_key": float(best_key),
        "trial_records": trial_records,
        **best_inner_audit,
    }
    return best_scores, best_cfg, audit


def main() -> None:
    parser = base_parser("Reviewer-facing PU-Bagging (bagging SVM) baseline")
    parser.add_argument("--max_trials", type=int, default=12)
    parser.add_argument("--n_bags", type=int, default=100)
    args = parser.parse_args()

    data, manifest = prepare_from_args(args)
    run_repeated_baseline(
        data=data,
        manifest=manifest,
        out_dir=args.out_dir,
        method_name=METHOD_NAME,
        method_spec=METHOD_SPEC,
        fit_score_fn=lambda d, m, s: fit_score(d, m, s, max_trials=args.max_trials, n_bags=args.n_bags),
        train_seed_base=args.train_seed_base,
        save_scores=not args.no_save_scores,
    )


if __name__ == "__main__":
    main()

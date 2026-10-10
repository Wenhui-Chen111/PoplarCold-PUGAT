# -*- coding: utf-8 -*-
"""Label Propagation / LLGC baseline (Zhou et al., NIPS 2003).

We use the exact weighted coexpression and PPI graphs from preprocessing, merged at
0.5/0.5 as a single affinity graph. No node features are used because the method is
network-only by design.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np
from scipy.sparse import coo_matrix, diags

from baseline_common import (
    BaselineData, SplitMasks, base_parser, compute_ranking_metrics,
    merge_weighted_networks, prepare_from_args, run_repeated_baseline,
    symmetrize_edges, validation_key,
)

METHOD_NAME = "LabelPropagation"
METHOD_SPEC: Dict[str, Any] = {
    "method": METHOD_NAME,
    "role": "network-only semi-supervised diffusion baseline",
    "algorithm": "Learning with Local and Global Consistency (Zhou et al., 2003).",
    "node_features": "None by method design.",
    "network_information": (
        "Exact preprocessed coexpression and PPI edge sets and weights. They are merged "
        "with fixed 0.5/0.5 source weights, symmetrized, and self-loops removed before diffusion."
    ),
    "training_labels": "Only manifest training positives are seeded with Y=1; all other nodes have initial Y=0 (unlabeled).",
    "loss": "No gradient loss; iterative fixed-point diffusion F_{t+1}=alpha*S*F_t+(1-alpha)*Y.",
    "sampling": "None.",
    "tuning": "Exhaustive alpha search on the fixed validation-positive + true-unlabeled universe.",
    "training_budget": "Each alpha is iterated to tolerance or max_iter (default 1000).",
    "transductive_note": "Held-out genes remain nodes in the full graph, consistent with the manuscript's transductive setting.",
}


def _build_normalized_affinity(data: BaselineData):
    edge_index, edge_weight = merge_weighted_networks(data, coexp_ratio=0.5, ppi_ratio=0.5)
    edge_index, edge_weight = symmetrize_edges(
        edge_index, edge_weight, data.n_nodes, remove_self_loops=True
    )
    W = coo_matrix(
        (edge_weight.astype(np.float64), (edge_index[0], edge_index[1])),
        shape=(data.n_nodes, data.n_nodes),
    ).tocsr()
    degree = np.asarray(W.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros_like(degree, dtype=np.float64)
    nz = degree > 0
    inv_sqrt[nz] = 1.0 / np.sqrt(degree[nz])
    D_inv_sqrt = diags(inv_sqrt)
    S = (D_inv_sqrt @ W @ D_inv_sqrt).tocsr()
    return S, int(W.nnz)


def _diffuse(S, train_pos: np.ndarray, alpha: float, max_iter: int, tol: float):
    y0 = train_pos.astype(np.float64)
    f = y0.copy()
    n_iter = 0
    final_delta = np.inf
    for i in range(1, max_iter + 1):
        new_f = alpha * (S @ f) + (1.0 - alpha) * y0
        final_delta = float(np.max(np.abs(new_f - f)))
        f = np.asarray(new_f).reshape(-1)
        n_iter = i
        if final_delta <= tol:
            break
    return f, n_iter, final_delta


def fit_score(
    data: BaselineData,
    masks: SplitMasks,
    seed: int,
    *,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
    del seed  # deterministic once graph/split are fixed
    S, nnz = _build_normalized_affinity(data)
    alpha_grid = [0.50, 0.70, 0.85, 0.90, 0.95, 0.97, 0.99]

    best_key = -np.inf
    best_scores = None
    best_cfg: Dict[str, Any] = {}
    best_iter = 0
    best_delta = np.nan
    trial_records = []

    for trial_id, alpha in enumerate(alpha_grid, start=1):
        scores, n_iter, delta = _diffuse(S, masks.train_pos, alpha, max_iter, tol)
        val_metrics = compute_ranking_metrics(scores, masks.val_pos, masks.true_unlabeled)
        key = validation_key(val_metrics)
        trial_records.append({
            "trial": trial_id, "alpha": alpha, "val_key": key,
            "iterations": n_iter, "final_delta": delta,
        })
        if key > best_key:
            best_key = key
            best_scores = scores.copy()
            best_cfg = {"alpha": alpha, "coexp_ratio": 0.5, "ppi_ratio": 0.5}
            best_iter = n_iter
            best_delta = delta

    if best_scores is None:
        raise RuntimeError("Label propagation search produced no result")

    audit = {
        "n_tuning_trials": len(alpha_grid),
        "validation_selection_key": "0.6*nDCG@100 + 0.4*Recall@100",
        "max_iter": int(max_iter),
        "tol": float(tol),
        "normalized_affinity_nnz": nnz,
        "best_iterations": int(best_iter),
        "best_final_delta": float(best_delta),
        "best_validation_key": float(best_key),
        "trial_records": trial_records,
    }
    return best_scores, best_cfg, audit


def main() -> None:
    parser = base_parser("Reviewer-facing Label Propagation baseline")
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-8)
    args = parser.parse_args()

    data, manifest = prepare_from_args(args)
    run_repeated_baseline(
        data=data,
        manifest=manifest,
        out_dir=args.out_dir,
        method_name=METHOD_NAME,
        method_spec=METHOD_SPEC,
        fit_score_fn=lambda d, m, s: fit_score(d, m, s, max_iter=args.max_iter, tol=args.tol),
        train_seed_base=args.train_seed_base,
        save_scores=not args.no_save_scores,
    )


if __name__ == "__main__":
    main()

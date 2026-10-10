# -*- coding: utf-8 -*-
"""Explicit EMOGI-style variant for the PoplarCold-PUGAT study.

Why "variant" rather than "EMOGI"
---------------------------------
Original EMOGI is a supervised/semi-supervised cancer-gene GCN built from a PPI
network, multi-omics node features, and curated positive/negative labels. This poplar
benchmark has positives and unlabeled genes but no verified negative set. Therefore we
do not claim to reproduce the original cancer labels. Instead we define a transparent
EMOGI-style adaptation:

* GCN message passing over one merged weighted biological graph;
* the exact X.npy node features used by PoplarCold-PUGAT;
* weighted BCE on training positives versus true-unlabeled presumed negatives;
* validation/test seed positives excluded from the loss;
* sigmoid output used as the ranking score.

This definition is deliberately explicit so the manuscript/rebuttal can state exactly
what "EMOGI variant" means in this study.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_common import (
    BaselineData, SplitMasks, base_parser, compute_ranking_metrics,
    merge_weighted_networks, prepare_from_args, run_repeated_baseline,
    set_seed, symmetrize_edges, validation_key,
)

METHOD_NAME = "EMOGI-variant"
METHOD_SPEC: Dict[str, Any] = {
    "method": METHOD_NAME,
    "role": "EMOGI-style single-graph GCN baseline adapted to the PU benchmark",
    "node_features": "Exact X.npy used by the main model.",
    "network_information": (
        "Exact coexpression and PPI source graphs/edge weights, merged into one weighted adjacency "
        "with fixed 0.5/0.5 source weights. This preserves the available network information while "
        "keeping the single-graph GCN structure of the EMOGI-style comparator."
    ),
    "training_labels": (
        "Training positives are class 1. y.npy==0 true-unlabeled genes are treated as presumed class 0 "
        "because this poplar benchmark has no curated negative set. Validation/test positives are masked out of the loss."
    ),
    "loss": "Weighted binary cross-entropy with pos_weight=n_true_unlabeled/n_train_positive.",
    "sampling": "No node sampling; full-batch transductive GCN training.",
    "tuning": (
        "12 configurations by default: hidden widths x learning rate x dropout. "
        "All use the same validation key 0.6*nDCG@100 + 0.4*Recall@100."
    ),
    "training_budget": (
        "Default tuning: 12 configurations x 200 optimizer steps. Final selected configuration: "
        "1000 optimizer steps, with validation used only for checkpoint selection (no early stopping)."
    ),
    "transductive_note": "Held-out nodes, their features and graph neighborhoods remain available; their positive labels are not used in training.",
}


class WeightedGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        support = self.linear(x)
        out = torch.zeros_like(support)
        out.index_add_(0, dst, support[src] * norm.unsqueeze(-1))
        return out


class EMOGIStyleGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims, 1]
        self.layers = nn.ModuleList([
            WeightedGCNLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, src, dst, norm)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h.squeeze(-1)


def _normalized_graph_tensors(data: BaselineData, device: torch.device):
    edge_index, edge_weight = merge_weighted_networks(data, 0.5, 0.5)
    edge_index, edge_weight = symmetrize_edges(edge_index, edge_weight, data.n_nodes, remove_self_loops=True)

    # Kipf-Welling style A+I normalization. Self-loop weight is 1.
    loops = np.arange(data.n_nodes, dtype=np.int64)
    src_np = np.concatenate([edge_index[0], loops])
    dst_np = np.concatenate([edge_index[1], loops])
    w_np = np.concatenate([edge_weight.astype(np.float64), np.ones(data.n_nodes, dtype=np.float64)])

    degree = np.bincount(src_np, weights=w_np, minlength=data.n_nodes).astype(np.float64)
    degree[degree <= 0] = 1.0
    norm_np = w_np / np.sqrt(degree[src_np] * degree[dst_np])

    src = torch.tensor(src_np, dtype=torch.long, device=device)
    dst = torch.tensor(dst_np, dtype=torch.long, device=device)
    norm = torch.tensor(norm_np, dtype=torch.float32, device=device)
    return src, dst, norm


def _train_once(
    data: BaselineData,
    masks: SplitMasks,
    seed: int,
    cfg: Mapping[str, Any],
    *,
    epochs: int,
    eval_every: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(data.X, dtype=torch.float32, device=device)
    src, dst, norm = _normalized_graph_tensors(data, device)

    train_pos = torch.tensor(masks.train_pos, dtype=torch.bool, device=device)
    true_unlabeled = torch.tensor(masks.true_unlabeled, dtype=torch.bool, device=device)
    train_mask = train_pos | true_unlabeled
    target = train_pos.float()
    n_pos = int(train_pos.sum().item())
    n_un = int(true_unlabeled.sum().item())
    pos_weight = float(n_un / max(n_pos, 1))

    model = EMOGIStyleGCN(
        in_dim=data.X.shape[1],
        hidden_dims=tuple(int(v) for v in cfg["hidden_dims"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
    )
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    best_key = -np.inf
    best_epoch = 0
    best_state = None
    best_scores = None
    last_loss = np.nan

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, src, dst, norm)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], target[train_mask], pos_weight=pos_weight_t
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = float(loss.item())

        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                scores = torch.sigmoid(model(x, src, dst, norm)).detach().cpu().numpy()
            val_metrics = compute_ranking_metrics(scores, masks.val_pos, masks.true_unlabeled)
            key = validation_key(val_metrics)
            if key > best_key:
                best_key = key
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_scores = scores.copy()

    if best_state is None or best_scores is None:
        raise RuntimeError("EMOGI-style GCN failed to produce a checkpoint")
    model.load_state_dict(best_state)

    audit = {
        "epochs_executed": int(epochs),
        "best_epoch": int(best_epoch),
        "best_validation_key": float(best_key),
        "last_training_loss": float(last_loss),
        "n_train_positive": n_pos,
        "n_presumed_negative": n_un,
        "pos_weight": pos_weight,
        "device": str(device),
    }
    return best_scores, audit


def fit_score(
    data: BaselineData,
    masks: SplitMasks,
    seed: int,
    *,
    max_trials: int = 12,
    tune_epochs: int = 200,
    final_epochs: int = 1000,
    eval_every: int = 10,
) -> Tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
    search_space = []
    for hidden_dims in [(64, 32), (100, 50), (128, 64)]:
        for lr in [1e-3, 3e-4]:
            for dropout in [0.3, 0.5]:
                search_space.append({
                    "hidden_dims": hidden_dims,
                    "lr": lr,
                    "dropout": dropout,
                    "weight_decay": 5e-4,
                })
    search_space = search_space[:max_trials]

    best_key = -np.inf
    selected: Dict[str, Any] = {}
    tuning_records = []
    for trial_id, cfg in enumerate(search_space, start=1):
        scores, train_audit = _train_once(
            data, masks, seed, cfg, epochs=tune_epochs, eval_every=eval_every
        )
        val_metrics = compute_ranking_metrics(scores, masks.val_pos, masks.true_unlabeled)
        key = validation_key(val_metrics)
        tuning_records.append({
            "trial": trial_id,
            "config": dict(cfg),
            "val_key": float(key),
            "best_epoch_within_trial": train_audit["best_epoch"],
        })
        if key > best_key:
            best_key = key
            selected = dict(cfg)

    if not selected:
        raise RuntimeError("EMOGI-style hyperparameter search produced no configuration")

    final_scores, final_audit = _train_once(
        data, masks, seed, selected, epochs=final_epochs, eval_every=eval_every
    )
    selected_out = dict(selected)
    selected_out.update({
        "tune_epochs": int(tune_epochs),
        "final_epochs": int(final_epochs),
        "eval_every": int(eval_every),
        "coexp_ratio": 0.5,
        "ppi_ratio": 0.5,
    })
    audit = {
        "n_tuning_trials": len(search_space),
        "tuning_optimizer_steps": int(len(search_space) * tune_epochs),
        "final_optimizer_steps": int(final_epochs),
        "validation_selection_key": "0.6*nDCG@100 + 0.4*Recall@100",
        "best_tuning_validation_key": float(best_key),
        "tuning_records": tuning_records,
        "final_training": final_audit,
    }
    return final_scores, selected_out, audit


def main() -> None:
    parser = base_parser("Reviewer-facing EMOGI-style GCN variant")
    parser.add_argument("--max_trials", type=int, default=12)
    parser.add_argument("--tune_epochs", type=int, default=200)
    parser.add_argument("--final_epochs", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=10)
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
            tune_epochs=args.tune_epochs,
            final_epochs=args.final_epochs,
            eval_every=args.eval_every,
        ),
        train_seed_base=args.train_seed_base,
        save_scores=not args.no_save_scores,
    )


if __name__ == "__main__":
    main()

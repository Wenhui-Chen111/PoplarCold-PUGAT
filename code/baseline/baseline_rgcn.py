# -*- coding: utf-8 -*-
"""Weighted R-GCN baseline for PoplarCold-PUGAT study.

The two biological networks are kept as separate relation types. Unlike the previous
script, edge weights are not discarded. The encoder is trained with the same nnPU risk
used by the main model's classification head, but without the main model's GATv2
attention, node-level fusion gate, pairwise ranking head, restart procedure, or MC
uncertainty. This makes R-GCN a focused multi-relational architecture comparator.
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
    prepare_from_args, run_repeated_baseline, set_seed, symmetrize_edges,
    validation_key,
)

METHOD_NAME = "R-GCN"
METHOD_SPEC: Dict[str, Any] = {
    "method": METHOD_NAME,
    "role": "multi-relational graph baseline",
    "node_features": "Exact X.npy used by the main model.",
    "network_information": (
        "Exact preprocessed coexpression and PPI graphs kept as two relation types; original edge weights are used."
    ),
    "training_labels": "Manifest training positives are P; y.npy==0 genes are U. Validation/test positives are excluded from the nnPU risk.",
    "loss": (
        "Non-negative PU logistic risk with the same working class prior pi as the main model. "
        "No pairwise ranking loss is used."
    ),
    "sampling": "No node or edge sampling; full-batch transductive training.",
    "tuning": (
        "12 configurations by default: hidden widths x learning rate x dropout, selected by "
        "0.6*nDCG@100 + 0.4*Recall@100 on the fixed validation universe."
    ),
    "training_budget": (
        "Default tuning: 12 configurations x 200 optimizer steps. Final selected configuration: "
        "1000 optimizer steps, validation only for checkpoint selection; no early stopping."
    ),
    "transductive_note": "Held-out nodes/features/edges remain in the graph; held-out positive labels are not used in the loss.",
}


def nnpu_loss(logits: torch.Tensor, pos_mask: torch.Tensor, unlabeled_mask: torch.Tensor, pi: float) -> torch.Tensor:
    positive_loss = F.softplus(-logits)
    negative_loss = F.softplus(logits)
    if int(pos_mask.sum().item()) == 0:
        return negative_loss[unlabeled_mask].mean()
    rp_pos = positive_loss[pos_mask].mean()
    rp_neg = negative_loss[pos_mask].mean()
    ru_neg = negative_loss[unlabeled_mask].mean()
    negative_risk = torch.clamp(ru_neg - float(pi) * rp_neg, min=0.0)
    return float(pi) * rp_pos + negative_risk


def _relation_tensors(edge_index: np.ndarray, edge_weight: np.ndarray, n_nodes: int, device: torch.device):
    edge_index, edge_weight = symmetrize_edges(
        edge_index, edge_weight, n_nodes=n_nodes, remove_self_loops=True
    )
    src_np = edge_index[0]
    dst_np = edge_index[1]
    w_np = edge_weight.astype(np.float64)
    # R-GCN normalization c_{i,r}: weighted incoming degree for each relation.
    degree = np.bincount(dst_np, weights=w_np, minlength=n_nodes).astype(np.float64)
    degree[degree <= 0] = 1.0
    norm_np = w_np / degree[dst_np]
    return (
        torch.tensor(src_np, dtype=torch.long, device=device),
        torch.tensor(dst_np, dtype=torch.long, device=device),
        torch.tensor(norm_np, dtype=torch.float32, device=device),
    )


class WeightedRGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_relations: int = 2) -> None:
        super().__init__()
        self.rel_linears = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(n_relations)
        ])
        self.self_linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, relations) -> torch.Tensor:
        out = self.self_linear(x)
        for linear, (src, dst, norm) in zip(self.rel_linears, relations):
            msg = linear(x[src]) * norm.unsqueeze(-1)
            out.index_add_(0, dst, msg)
        return out


class WeightedRGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims]
        self.layers = nn.ModuleList([
            WeightedRGCNLayer(dims[i], dims[i + 1], n_relations=2)
            for i in range(len(dims) - 1)
        ])
        last_dim = dims[-1] if len(dims) > 1 else in_dim
        self.head = nn.Linear(last_dim, 1)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, relations) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, relations)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h).squeeze(-1)


def _train_once(
    data: BaselineData,
    masks: SplitMasks,
    seed: int,
    cfg: Mapping[str, Any],
    *,
    pi: float,
    epochs: int,
    eval_every: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(data.X, dtype=torch.float32, device=device)
    relations = [
        _relation_tensors(data.coexp_edge_index, data.coexp_edge_weight, data.n_nodes, device),
        _relation_tensors(data.ppi_edge_index, data.ppi_edge_weight, data.n_nodes, device),
    ]
    pos_mask = torch.tensor(masks.train_pos, dtype=torch.bool, device=device)
    unlabeled_mask = torch.tensor(masks.true_unlabeled, dtype=torch.bool, device=device)

    model = WeightedRGCN(
        in_dim=data.X.shape[1],
        hidden_dims=tuple(int(v) for v in cfg["hidden_dims"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
    )

    best_key = -np.inf
    best_epoch = 0
    best_state = None
    best_scores = None
    last_loss = np.nan

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, relations)
        loss = nnpu_loss(logits, pos_mask, unlabeled_mask, pi=pi)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = float(loss.item())

        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                scores = torch.sigmoid(model(x, relations)).detach().cpu().numpy()
            val_metrics = compute_ranking_metrics(scores, masks.val_pos, masks.true_unlabeled)
            key = validation_key(val_metrics)
            if key > best_key:
                best_key = key
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_scores = scores.copy()

    if best_state is None or best_scores is None:
        raise RuntimeError("R-GCN failed to produce a checkpoint")
    model.load_state_dict(best_state)

    audit = {
        "epochs_executed": int(epochs),
        "best_epoch": int(best_epoch),
        "best_validation_key": float(best_key),
        "last_training_loss": float(last_loss),
        "pi": float(pi),
        "n_train_positive": int(masks.train_pos.sum()),
        "n_true_unlabeled": int(masks.true_unlabeled.sum()),
        "coexp_edges": int(data.coexp_edge_index.shape[1]),
        "ppi_edges": int(data.ppi_edge_index.shape[1]),
        "device": str(device),
    }
    return best_scores, audit


def fit_score(
    data: BaselineData,
    masks: SplitMasks,
    seed: int,
    *,
    pi: float = 0.013,
    max_trials: int = 12,
    tune_epochs: int = 200,
    final_epochs: int = 1000,
    eval_every: int = 10,
) -> Tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
    if not 0.0 < pi < 1.0:
        raise ValueError("pi must be in (0, 1)")

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
            data, masks, seed, cfg, pi=pi, epochs=tune_epochs, eval_every=eval_every
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
        raise RuntimeError("R-GCN hyperparameter search produced no configuration")

    final_scores, final_audit = _train_once(
        data, masks, seed, selected, pi=pi, epochs=final_epochs, eval_every=eval_every
    )
    selected_out = dict(selected)
    selected_out.update({
        "pi": float(pi),
        "tune_epochs": int(tune_epochs),
        "final_epochs": int(final_epochs),
        "eval_every": int(eval_every),
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
    parser = base_parser("Reviewer-facing weighted R-GCN baseline")
    parser.add_argument("--pi", type=float, default=0.013,
                        help="Use the same working PU prior as the main model")
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
            pi=args.pi,
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

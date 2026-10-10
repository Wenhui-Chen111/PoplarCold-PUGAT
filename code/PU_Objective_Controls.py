#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PU_Objective_Controls.py

Reviewer-facing objective controls for PoplarCold-PUGAT.

This script intentionally reuses the revised main training code for:
  * data loading and graph construction
  * split-manifest validation and masks
  * the PoplarCold-PUGAT architecture
  * the original nnPU component
  * the original high-scoring-unlabeled ranking sampler
  * evaluation metrics and MC-dropout prediction
  * bootstrap CI utilities

It adds only the controls needed to isolate the training objective:
  1) full_highscore       : nnPU(Z) + ranking(R), high-scoring U sampling
  2) bce_highscore        : matched BCE(Z) + ranking(R), high-scoring U sampling
  3) ranking_only_highscore: ranking(R) only, high-scoring U sampling
  4) nnpu_only            : nnPU(Z) only; IMPORTANT: evaluation uses trained Z output
  5) full_uniform         : nnPU(Z) + ranking(R), uniform U sampling

All variants use the same split, architecture, initialization seed, graph inputs,
optimizer settings, warm-restart schedule, and (by default) exact optimizer-step budget.

The positive-class prior supplied by --pi is treated as a WORKING HYPERPARAMETER,
not as an estimate of biological cold-associated-gene prevalence.

Typical use with the repository-relative defaults:

python PU_Objective_Controls.py \
  --main_script Train_ColdPU.py \
  --data_dir data/preprocessed_label_independent_ppi800_corr0.75 \
  --pi 0.013

For a short smoke test, for example:

python PU_Objective_Controls.py --max_repeats 1 --warm_restart_stages 0 \
  --stage_epochs 2 --val_mc_times 2 --final_mc_times 2 --bootstrap_reps 200
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_MAIN_SCRIPT = str(REPO_DIR / "Train_ColdPU.py")
DEFAULT_DATA_DIR = str(REPO_DIR / "data" / "preprocessed_label_independent_ppi800_corr0.75")
DEFAULT_MANIFESTS = [
    f"random={REPO_DIR / 'results' / 'random' / 'manifest' / 'split_manifest.csv'}",
    f"grouped={REPO_DIR / 'results' / 'grouped' / 'manifest' / 'split_manifest.csv'}",
]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    cls_objective: str      # nnpu | bce | none
    rank_objective: bool
    rank_sampler: str       # highscore | uniform | none
    primary_output: str     # rank_score | pu_score
    reviewer_role: str


VARIANTS: Dict[str, VariantSpec] = {
    "full_highscore": VariantSpec(
        "full_highscore", "nnpu", True, "highscore", "rank_score",
        "Full model: nnPU on Z + ranking loss on R with high-scoring unlabeled sampling",
    ),
    "bce_highscore": VariantSpec(
        "bce_highscore", "bce", True, "highscore", "rank_score",
        "Matched BCE control: BCE on Z (unlabeled treated as negative) + same ranking loss on R",
    ),
    "ranking_only_highscore": VariantSpec(
        "ranking_only_highscore", "none", True, "highscore", "rank_score",
        "Ranking-only control: no classification/PU term; R head trained by pairwise ranking only",
    ),
    "nnpu_only": VariantSpec(
        "nnpu_only", "nnpu", False, "none", "pu_score",
        "nnPU-only control: Z head is trained; evaluation uses sigmoid(Z), not the untrained R head",
    ),
    "full_uniform": VariantSpec(
        "full_uniform", "nnpu", True, "uniform", "rank_score",
        "Sampling control: same full objective but unlabeled ranking partners are sampled uniformly",
    ),
}


PRIMARY_METRICS = ["nDCG@100", "Recall@100", "AUPRC", "AUROC"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reviewer-facing controls that isolate nnPU, ranking, BCE, and unlabeled-sampling effects."
    )
    p.add_argument("--main_script", type=str, default=DEFAULT_MAIN_SCRIPT,
                   help="Path to Train_ColdPU.py. A local sibling fallback is tried if this path is absent.")
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--manifest", action="append", default=[], metavar="SCHEME=PATH",
                   help=("Repeatable split manifest. If omitted, repository-relative random and grouped manifests are used. "
                         "Example: --manifest random=/path/split_manifest.csv"))
    p.add_argument("--save_root", type=str, default="")
    p.add_argument("--variants", type=str,
                   default="full_highscore,bce_highscore,ranking_only_highscore,nnpu_only,full_uniform",
                   help="Comma-separated subset of objective controls")
    p.add_argument("--max_repeats", type=int, default=0,
                   help="0 = use every repeat in each manifest; positive value is useful for smoke tests")

    # Positive-class prior: reviewer-facing semantics are explicit.
    p.add_argument("--pi", type=float, default=0.013,
                   help=("Working nnPU class-prior hyperparameter. This script does NOT interpret it as biological prevalence."))
    p.add_argument("--beta", type=float, default=0.0,
                   help="nnPU negative-risk lower bound; 0 reproduces non-negative PU")

    # Model: defaults match revised main model.
    p.add_argument("--graph_mode", choices=["hetero", "merged"], default="hetero")
    p.add_argument("--coexp_ratio", type=float, default=0.5)
    p.add_argument("--ppi_ratio", type=float, default=0.5)
    p.add_argument("--no_coexp", action="store_true")
    p.add_argument("--no_ppi", action="store_true")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--proj_dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)

    # Ranking loss.
    p.add_argument("--rank_weight", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--num_rank_pairs", type=int, default=2048)
    p.add_argument("--hard_negative_ratio", type=float, default=0.5,
                   help="Top fraction of unlabeled R scores forming the high-scoring pool")
    p.add_argument("--bce_mode", choices=["balanced", "unweighted", "prior_weighted"], default="balanced",
                   help=("Matched BCE definition. balanced gives equal mean weight to observed P and U groups; "
                         "unweighted pools all P/U examples; prior_weighted uses pi and 1-pi group weights."))

    # Equal training budget.
    p.add_argument("--seed", type=int, default=42,
                   help="Global utility seed; split assignments themselves come from the supplied manifests")
    p.add_argument("--train_seed_base", type=int, default=10042)
    p.add_argument("--train_seed_mode", choices=["independent", "same_as_split"], default="independent")
    p.add_argument("--warm_restart_stages", type=int, default=4,
                   help="Number of optimizer restart stages after the base stage; total stages = 1 + this value")
    p.add_argument("--stage_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--budget_mode", choices=["fixed", "early_stop"], default="fixed",
                   help=("fixed runs the exact same optimizer-step budget for every variant; "
                         "early_stop is provided only for legacy compatibility."))
    p.add_argument("--patience_checks", type=int, default=8,
                   help="Only used in early_stop mode; counted in validation checks, not epochs")
    p.add_argument("--val_mc_times", type=int, default=10)
    p.add_argument("--final_mc_times", type=int, default=30)
    p.add_argument("--uncertainty_penalty", type=float, default=0.10)
    p.add_argument("--hidden_seed_as_unlabeled", action="store_true",
                   help="Pass the revised main-code hidden-seed robustness setting into every objective control")

    # Reviewer-facing inference/statistics.
    p.add_argument("--bootstrap_reps", type=int, default=10000)
    p.add_argument("--save_per_repeat_rankings", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--topk_overlap", type=str, default="50,100,200")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


def resolve_main_script(requested: str) -> Path:
    candidates = [Path(requested)]
    here = Path(__file__).resolve().parent
    candidates.extend([
        here / "Train_ColdPU.py",
        Path.cwd() / "Train_ColdPU.py",
    ])
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "Could not find main script. Tried:\n" + "\n".join(str(x) for x in candidates)
    )


def import_main_module(path: Path):
    spec = importlib.util.spec_from_file_location("coldpu_main", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import main script from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = [
        "set_seed", "load_preprocessed", "validate_split_manifest", "build_data_for_split",
        "SingleOmicPUGAT", "PULossWithRanking", "compute_ranking_metrics",
        "mc_dropout_predict", "bootstrap_mean_ci", "summarize_metrics_df",
    ]
    missing = [x for x in required if not hasattr(module, x)]
    if missing:
        raise AttributeError(f"Main script is missing required reusable definitions: {missing}")
    return module


def parse_manifest_specs(specs: List[str]) -> List[Tuple[str, Path]]:
    if not specs:
        specs = list(DEFAULT_MANIFESTS)
    out: List[Tuple[str, Path]] = []
    seen = set()
    for raw in specs:
        if "=" not in raw:
            raise ValueError(f"--manifest must be SCHEME=PATH, got: {raw}")
        scheme, path = raw.split("=", 1)
        scheme = scheme.strip()
        path = Path(path.strip())
        if not scheme:
            raise ValueError(f"Empty manifest scheme in: {raw}")
        if scheme in seen:
            raise ValueError(f"Duplicate manifest scheme: {scheme}")
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        seen.add(scheme)
        out.append((scheme, path.resolve()))
    return out


def selected_variants(text: str) -> List[VariantSpec]:
    names = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in names if x not in VARIANTS]
    if bad:
        raise ValueError(f"Unknown variant(s): {bad}; valid={list(VARIANTS)}")
    if not names:
        raise ValueError("No variants selected")
    return [VARIANTS[x] for x in names]


def choose_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scalarize_stats(stats: Dict[str, object]) -> Dict[str, float]:
    out = {}
    for k, v in stats.items():
        if torch.is_tensor(v):
            out[k] = float(v.detach().cpu().item())
        elif isinstance(v, (float, int, np.floating, np.integer)):
            out[k] = float(v)
    return out


class ObjectiveControlLoss(nn.Module):
    """Control objective that reuses nnPU/high-score ranking code from the main script."""

    def __init__(self, core, spec: VariantSpec, pi: float, beta: float,
                 rank_weight: float, margin: float, num_rank_pairs: int,
                 hard_negative_ratio: float, bce_mode: str):
        super().__init__()
        self.core = core
        self.spec = spec
        self.pi = float(pi)
        self.rank_weight = float(rank_weight)
        self.bce_mode = str(bce_mode)
        # Main-code helper guarantees that nnPU and high-score sampling are not reimplemented differently.
        self.helper = core.PULossWithRanking(
            pi=pi, beta=beta, rank_weight=rank_weight,
            margin=margin, num_rank_pairs=num_rank_pairs,
            hard_negative_ratio=hard_negative_ratio,
        )
        self.margin = float(margin)
        self.num_rank_pairs = int(num_rank_pairs)

    def _matched_bce(self, logits, pos_mask, unlabeled_mask):
        pos_logits = logits[pos_mask]
        u_logits = logits[unlabeled_mask]
        if pos_logits.numel() == 0 or u_logits.numel() == 0:
            raise RuntimeError("BCE control requires non-empty train positive and training-unlabeled sets")
        pos_loss = F.softplus(-pos_logits).mean()  # target=1
        u_neg_loss = F.softplus(u_logits).mean()  # target=0 for the P-vs-U BCE control

        if self.bce_mode == "balanced":
            loss = 0.5 * pos_loss + 0.5 * u_neg_loss
        elif self.bce_mode == "unweighted":
            total = pos_logits.numel() + u_logits.numel()
            loss = (F.softplus(-pos_logits).sum() + F.softplus(u_logits).sum()) / float(total)
        elif self.bce_mode == "prior_weighted":
            loss = self.pi * pos_loss + (1.0 - self.pi) * u_neg_loss
        else:
            raise ValueError(self.bce_mode)
        return loss, {
            "bce_pos": pos_loss.detach(),
            "bce_u_as_neg": u_neg_loss.detach(),
            "bce_loss": loss.detach(),
        }

    def _uniform_ranking(self, rank_logits, pos_mask, unlabeled_mask):
        pos_idx = torch.where(pos_mask)[0]
        u_idx = torch.where(unlabeled_mask)[0]
        if pos_idx.numel() == 0 or u_idx.numel() == 0:
            return torch.tensor(0.0, device=rank_logits.device)
        pos_sample = pos_idx[torch.randint(0, pos_idx.numel(), (self.num_rank_pairs,), device=rank_logits.device)]
        u_sample = u_idx[torch.randint(0, u_idx.numel(), (self.num_rank_pairs,), device=rank_logits.device)]
        return F.relu(self.margin - (rank_logits[pos_sample] - rank_logits[u_sample])).mean()

    def forward(self, final_logit, rank_logit, pos_mask, unlabeled_mask):
        zero = final_logit.sum() * 0.0
        stats: Dict[str, torch.Tensor] = {}

        if self.spec.cls_objective == "nnpu":
            cls_loss, pu_stats = self.helper.nnpu_part(final_logit, pos_mask, unlabeled_mask)
            stats.update(pu_stats)
            stats["nnpu_loss"] = cls_loss.detach()
        elif self.spec.cls_objective == "bce":
            cls_loss, bce_stats = self._matched_bce(final_logit, pos_mask, unlabeled_mask)
            stats.update(bce_stats)
        elif self.spec.cls_objective == "none":
            cls_loss = zero
        else:
            raise ValueError(self.spec.cls_objective)

        if self.spec.rank_objective:
            if self.spec.rank_sampler == "highscore":
                # Exact high-scoring sampler from revised main model.
                rank_loss = self.helper.ranking_part(rank_logit, pos_mask, unlabeled_mask)
            elif self.spec.rank_sampler == "uniform":
                rank_loss = self._uniform_ranking(rank_logit, pos_mask, unlabeled_mask)
            else:
                raise ValueError(self.spec.rank_sampler)
        else:
            rank_loss = rank_logit.sum() * 0.0

        total = cls_loss + self.rank_weight * rank_loss
        stats["classification_loss"] = cls_loss.detach()
        stats["rank_loss"] = rank_loss.detach()
        stats["total_loss"] = total.detach()
        return total, stats


def model_builder(core, data, args):
    return core.SingleOmicPUGAT(
        in_dim=data.x.size(1),
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        heads=args.heads,
        dropout=args.dropout,
        use_hetero=(args.graph_mode == "hetero"),
    )


def predict_for_variant(core, model, data, spec: VariantSpec,
                        mc_times: int, uncertainty_penalty: float):
    preds = core.mc_dropout_predict(
        model, data, mc_times=mc_times, uncertainty_penalty=uncertainty_penalty
    )
    if spec.primary_output == "pu_score":
        primary = np.asarray(preds["pu_score"], dtype=float)
    elif spec.primary_output == "rank_score":
        primary = np.asarray(preds["rank_score"], dtype=float)
    else:
        raise ValueError(spec.primary_output)
    return preds, primary


def train_variant(core, model, data, spec: VariantSpec, args, device: torch.device,
                  run_dir: Path, run_tag: str) -> Tuple[nn.Module, pd.DataFrame, Dict, Dict, Dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    data = data.to(device)
    criterion = ObjectiveControlLoss(
        core=core, spec=spec, pi=args.pi, beta=args.beta,
        rank_weight=args.rank_weight, margin=args.margin,
        num_rank_pairs=args.num_rank_pairs,
        hard_negative_ratio=args.hard_negative_ratio,
        bce_mode=args.bce_mode,
    ).to(device)

    history: List[Dict] = []
    global_best_state = None
    global_best_val = -np.inf
    global_best_stage = -1
    global_best_epoch = -1
    optimizer_steps = 0
    validation_checks = 0
    stopped_early_any_stage = False

    n_stages = 1 + int(args.warm_restart_stages)
    for stage in range(n_stages):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=20, min_lr=1e-5
        )
        stage_best_state = None
        stage_best_val = -np.inf
        bad_checks = 0

        for epoch in range(1, args.stage_epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(data, return_details=True)
            loss, loss_stats = criterion(
                out["final_logit"], out["rank_logit"],
                data.train_pos_mask, data.train_unlabeled_mask,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss for {spec.name}, stage={stage}, epoch={epoch}: {loss.item()}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer_steps += 1

            do_eval = (epoch == 1 or epoch % args.log_steps == 0 or epoch == args.stage_epochs)
            if not do_eval:
                continue

            model.eval()
            preds, primary = predict_for_variant(
                core, model, data, spec,
                mc_times=args.val_mc_times,
                uncertainty_penalty=args.uncertainty_penalty,
            )
            val_m = core.compute_ranking_metrics(
                primary,
                eval_pos_mask=data.val_pos_mask,
                unlabeled_mask=data.unlabeled_mask,
                Ks=(50, 100, 200),
            )
            val_key = 0.6 * val_m.get("nDCG@100", 0.0) + 0.4 * val_m.get("Recall@100", 0.0)
            scheduler.step(val_key)
            validation_checks += 1

            rec = {
                "stage": int(stage), "epoch": int(epoch),
                "variant": spec.name,
                "primary_output": spec.primary_output,
                "val_key": float(val_key),
                "lr": float(optimizer.param_groups[0]["lr"]),
                **scalarize_stats(loss_stats),
                **{f"val_{k}": float(v) for k, v in val_m.items()},
            }
            history.append(rec)

            if val_key > stage_best_val:
                stage_best_val = float(val_key)
                stage_best_state = copy.deepcopy(model.state_dict())
                bad_checks = 0
            else:
                bad_checks += 1

            if val_key > global_best_val:
                global_best_val = float(val_key)
                global_best_stage = int(stage)
                global_best_epoch = int(epoch)
                global_best_state = copy.deepcopy(model.state_dict())
                torch.save(global_best_state, run_dir / f"{run_tag}_best_global.pth")

            if args.budget_mode == "early_stop" and bad_checks >= args.patience_checks:
                stopped_early_any_stage = True
                break

        # Reproduce the main script's warm-restart behavior: next stage starts from
        # the best checkpoint of the current stage, not simply the final epoch.
        if stage_best_state is not None:
            model.load_state_dict(stage_best_state)

    if global_best_state is None:
        raise RuntimeError(f"No checkpoint was selected for {run_tag}")
    model.load_state_dict(global_best_state)

    model.eval()
    final_preds, primary = predict_for_variant(
        core, model, data, spec,
        mc_times=args.final_mc_times,
        uncertainty_penalty=args.uncertainty_penalty,
    )
    test_m = core.compute_ranking_metrics(
        primary,
        eval_pos_mask=data.test_pos_mask,
        unlabeled_mask=data.unlabeled_mask,
        Ks=(50, 100, 200),
    )

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(run_dir / f"{run_tag}_training_history.csv", index=False)

    budget = {
        "optimizer_steps": int(optimizer_steps),
        "expected_fixed_optimizer_steps": int(n_stages * args.stage_epochs),
        "validation_checks": int(validation_checks),
        "best_val_key": float(global_best_val),
        "best_stage": int(global_best_stage),
        "best_epoch": int(global_best_epoch),
        "stopped_early_any_stage": bool(stopped_early_any_stage),
        "n_stages": int(n_stages),
        "stage_epochs": int(args.stage_epochs),
    }
    return model, hist_df, test_m, final_preds, budget


def summarize_by_variant(core, per_split: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    id_cols = ["repeat", "split_seed", "train_seed"]
    metric_cols = [
        c for c in per_split.columns
        if c in {"MRR", "AUPRC", "AUROC"}
        or c.startswith("Hits@") or c.startswith("Precision@")
        or c.startswith("Recall@") or c.startswith("nDCG@")
    ]
    for (scheme, variant), sub in per_split.groupby(["split_scheme", "variant"], sort=False):
        tmp = sub[id_cols + metric_cols].copy()
        summary = core.summarize_metrics_df(
            tmp, ci=0.95, n_boot=args.bootstrap_reps,
            seed=args.seed + 100000 + sum(ord(ch) for ch in f"{scheme}:{variant}"),
        )
        summary.insert(0, "variant", variant)
        summary.insert(0, "split_scheme", scheme)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def make_primary_reviewer_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scheme, variant), sub in summary.groupby(["split_scheme", "variant"], sort=False):
        row = {"split_scheme": scheme, "variant": variant}
        for metric in PRIMARY_METRICS:
            hit = sub.loc[sub["metric"] == metric]
            if len(hit) == 1:
                r = hit.iloc[0]
                row[f"{metric}_mean"] = r["mean"]
                row[f"{metric}_std"] = r["std"]
                row[f"{metric}_ci95_low"] = r["ci95_boot_low"]
                row[f"{metric}_ci95_high"] = r["ci95_boot_high"]
                row[f"{metric}_n"] = r["n_repeats"]
        rows.append(row)
    return pd.DataFrame(rows)


def paired_contrasts(core, per_split: pd.DataFrame, args) -> pd.DataFrame:
    contrasts = [
        ("nnPU_vs_BCE_with_ranking", "full_highscore", "bce_highscore",
         "Isolates nnPU vs treating U as negatives by matched BCE; same R ranking objective/sampling."),
        ("nnPU_added_to_ranking", "full_highscore", "ranking_only_highscore",
         "Isolates the added nnPU term beyond the same ranking-only objective."),
        ("ranking_added_to_nnPU", "full_highscore", "nnpu_only",
         "Requested full vs nnPU-only control; nnPU-only is evaluated with trained Z output as requested."),
        ("highscore_vs_uniform_sampling", "full_highscore", "full_uniform",
         "Isolates high-scoring vs uniform unlabeled sampling with the full objective held fixed."),
    ]
    metric_cols = [
        c for c in per_split.columns
        if c in {"MRR", "AUPRC", "AUROC"}
        or c.startswith("Recall@") or c.startswith("nDCG@")
    ]
    rows = []
    test_counter = 0
    for scheme, sdata in per_split.groupby("split_scheme", sort=False):
        for cname, a, b, interpretation in contrasts:
            if a not in set(sdata["variant"]) or b not in set(sdata["variant"]):
                continue
            da = sdata.loc[sdata["variant"] == a].copy()
            db = sdata.loc[sdata["variant"] == b].copy()
            keys = ["repeat", "split_seed"]
            merged = da.merge(db, on=keys, suffixes=("_A", "_B"), how="inner")
            for metric in metric_cols:
                ca, cb = f"{metric}_A", f"{metric}_B"
                if ca not in merged or cb not in merged:
                    continue
                diff = pd.to_numeric(merged[ca], errors="coerce") - pd.to_numeric(merged[cb], errors="coerce")
                diff = diff[np.isfinite(diff)].to_numpy(dtype=float)
                if len(diff) == 0:
                    continue
                lo, hi = core.bootstrap_mean_ci(
                    diff, n_boot=args.bootstrap_reps,
                    seed=args.seed + 200000 + test_counter,
                )
                pval = np.nan
                if len(diff) >= 2 and getattr(core, "SCIPY_AVAILABLE", False) and getattr(core, "wilcoxon", None) is not None:
                    if np.allclose(diff, 0.0):
                        pval = 1.0
                    else:
                        try:
                            pval = float(core.wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
                        except Exception:
                            pval = np.nan
                rows.append({
                    "split_scheme": scheme,
                    "contrast": cname,
                    "variant_A": a,
                    "variant_B": b,
                    "difference_definition": "A_minus_B",
                    "metric": metric,
                    "n_paired_repeats": int(len(diff)),
                    "mean_difference": float(np.mean(diff)),
                    "median_difference": float(np.median(diff)),
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "wilcoxon_p": pval,
                    "reviewer_interpretation": interpretation,
                })
                test_counter += 1
    out = pd.DataFrame(rows)
    if len(out):
        if hasattr(core, "_holm_adjust"):
            out["holm_adjusted_p_all_contrasts"] = core._holm_adjust(out["wilcoxon_p"].tolist())
        else:
            out["holm_adjusted_p_all_contrasts"] = np.nan
    return out


def make_primary_contrast_table(contrasts: pd.DataFrame) -> pd.DataFrame:
    if contrasts.empty:
        return contrasts
    return contrasts.loc[contrasts["metric"].isin(PRIMARY_METRICS)].reset_index(drop=True)


def save_ensemble_rankings(ensemble_store: Dict[Tuple[str, str], List[np.ndarray]],
                           bundle: Dict, save_root: Path, topks: List[int]) -> pd.DataFrame:
    known = np.asarray(bundle["y"], dtype=np.int64) == 1
    nodes = np.asarray(bundle["nodes"], dtype=object)
    out_dir = save_root / "ensemble_rankings"
    out_dir.mkdir(exist_ok=True)
    ensemble_scores: Dict[Tuple[str, str], np.ndarray] = {}

    for key, arrays in ensemble_store.items():
        scheme, variant = key
        mat = np.vstack(arrays)
        mean_score = mat.mean(axis=0)
        sd_score = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros(mat.shape[1], dtype=float)
        ensemble_scores[key] = mean_score
        df = pd.DataFrame({
            "gene_id": nodes,
            "mean_primary_score": mean_score,
            "sd_primary_score_across_repeats": sd_score,
            "is_known_seed": known,
        })
        df = df.loc[~df["is_known_seed"]].sort_values("mean_primary_score", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        df.to_csv(out_dir / f"{scheme}__{variant}__ensemble_ranking.csv", index=False)

    overlap_rows = []
    for scheme in sorted({k[0] for k in ensemble_scores}):
        full_key = (scheme, "full_highscore")
        if full_key not in ensemble_scores:
            continue
        eligible = np.where(~known)[0]
        full_order = eligible[np.argsort(-ensemble_scores[full_key][eligible])]
        for (s, variant), scores in ensemble_scores.items():
            if s != scheme or variant == "full_highscore":
                continue
            other_order = eligible[np.argsort(-scores[eligible])]
            for k in topks:
                k_eff = min(k, len(eligible))
                a = set(full_order[:k_eff].tolist())
                b = set(other_order[:k_eff].tolist())
                inter = len(a & b)
                union = len(a | b)
                overlap_rows.append({
                    "split_scheme": scheme,
                    "reference_variant": "full_highscore",
                    "other_variant": variant,
                    "K": int(k),
                    "overlap_count": int(inter),
                    "overlap_fraction_of_K": float(inter / max(k_eff, 1)),
                    "jaccard": float(inter / union) if union else np.nan,
                })
    return pd.DataFrame(overlap_rows)


def main():
    args = parse_args()
    if not (0.0 < args.pi < 1.0):
        raise ValueError("--pi must lie strictly between 0 and 1")
    if not (0.0 < args.hard_negative_ratio <= 1.0):
        raise ValueError("--hard_negative_ratio must be in (0,1]")
    if args.stage_epochs <= 0 or args.warm_restart_stages < 0:
        raise ValueError("Invalid training budget")

    main_path = resolve_main_script(args.main_script)
    core = import_main_module(main_path)
    variants = selected_variants(args.variants)
    manifests = parse_manifest_specs(args.manifest)
    device = choose_device(args.device)
    core.device = str(device)
    core.set_seed(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = Path(args.save_root) if args.save_root else Path(
        REPO_DIR / "results" / f"pu_objective_controls_{ts}"
    )
    save_root.mkdir(parents=True, exist_ok=True)
    (save_root / "per_repeat").mkdir(exist_ok=True)

    config = vars(args).copy()
    config.update({
        "main_script_resolved": str(main_path),
        "device_resolved": str(device),
        "positive_class_prior_interpretation": "working_hyperparameter_not_biological_prevalence_estimate",
        "selected_variants": [v.name for v in variants],
        "manifest_specs_resolved": {scheme: str(path) for scheme, path in manifests},
        "total_stages": 1 + args.warm_restart_stages,
        "fixed_optimizer_steps_per_run": (1 + args.warm_restart_stages) * args.stage_epochs,
    })
    with open(save_root / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"[+] Main script: {main_path}")
    print(f"[+] Device: {device}")
    print(f"[+] Save root: {save_root}")
    print(f"[+] Working pi={args.pi:.6f} (explicit hyperparameter; NOT a biological prevalence estimate)")
    print(f"[+] Variants: {[v.name for v in variants]}")

    bundle = core.load_preprocessed(args.data_dir)
    y = np.asarray(bundle["y"], dtype=np.int64)
    n_nodes = int(len(y))
    n_seed = int((y == 1).sum())
    pos_genes = [bundle["nodes"][i] for i in np.where(y == 1)[0]]
    observed_seed_fraction = n_seed / max(n_nodes, 1)
    print(f"[i] Preprocessed candidate universe: nodes={n_nodes}, known seeds={n_seed}, seed fraction={observed_seed_fraction:.6f}")

    all_metric_rows: List[Dict] = []
    budget_rows: List[Dict] = []
    prior_rows: List[Dict] = []
    run_rows: List[Dict] = []
    ensemble_store: Dict[Tuple[str, str], List[np.ndarray]] = {}

    for scheme, manifest_path in manifests:
        manifest = pd.read_csv(manifest_path)
        manifest["repeat"] = pd.to_numeric(manifest["repeat"], errors="raise").astype(int)
        manifest["split_seed"] = pd.to_numeric(manifest["split_seed"], errors="raise").astype(int)
        manifest["gene_id"] = manifest["gene_id"].astype(str)
        manifest["partition"] = manifest["partition"].astype(str)
        validation_mode = "grouped" if "group" in scheme.lower() else "random"
        core.validate_split_manifest(manifest, pos_genes, split_mode=validation_mode)

        repeats = sorted(manifest["repeat"].unique().tolist())
        if args.max_repeats > 0:
            repeats = repeats[:args.max_repeats]
        print(f"\n[+] Split scheme '{scheme}': {len(repeats)} repeat(s) from {manifest_path}")

        for rep_index, rep in enumerate(repeats):
            mrep = manifest.loc[manifest["repeat"] == rep].copy()
            split_seeds = mrep["split_seed"].unique()
            if len(split_seeds) != 1:
                raise ValueError(f"{scheme} repeat {rep} has multiple split_seed values: {split_seeds}")
            split_seed = int(split_seeds[0])
            if args.train_seed_mode == "independent":
                train_seed = int(args.train_seed_base + int(rep) - 1)
            else:
                train_seed = split_seed

            # Build once per repeat, then deep-copy for each variant so masks/graphs are identical.
            data_cpu = core.build_data_for_split(
                bundle=bundle,
                manifest_repeat=mrep,
                hidden_seed_as_unlabeled=args.hidden_seed_as_unlabeled,
                use_coexp=not args.no_coexp,
                use_ppi=not args.no_ppi,
                coexp_ratio=args.coexp_ratio,
                ppi_ratio=args.ppi_ratio,
                use_hetero=(args.graph_mode == "hetero"),
            )
            n_train_pos = int(data_cpu.train_pos_mask.sum().item())
            n_val_pos = int(data_cpu.val_pos_mask.sum().item())
            n_test_pos = int(data_cpu.test_pos_mask.sum().item())
            n_train_u = int(data_cpu.train_unlabeled_mask.sum().item())
            prior_rows.append({
                "split_scheme": scheme,
                "repeat": int(rep),
                "split_seed": split_seed,
                "n_candidate_nodes": n_nodes,
                "n_known_seed_positive_after_preprocessing": n_seed,
                "observed_seed_fraction_all_nodes": observed_seed_fraction,
                "n_train_positive_labels": n_train_pos,
                "n_val_hidden_positive_labels": n_val_pos,
                "n_test_hidden_positive_labels": n_test_pos,
                "n_training_unlabeled_pool": n_train_u,
                "working_pi": float(args.pi),
                "pi_interpretation": "hyperparameter_not_biological_prevalence",
                "seed_ascertainment_note": "literature_GO_homology_based_seeds_are_selected_not_random_positive_sample",
            })

            for spec in variants:
                # Reset to exactly the same initialization/training RNG seed for every
                # objective variant within a repeat. This makes the objective comparison paired.
                core.set_seed(train_seed)
                data = copy.deepcopy(data_cpu)
                model = model_builder(core, data, args)
                run_tag = f"{scheme}__repeat_{int(rep):02d}__{spec.name}"
                run_dir = save_root / "per_repeat" / run_tag
                print(
                    f"\n--- {run_tag} | split_seed={split_seed} train_seed={train_seed} "
                    f"| primary={spec.primary_output} ---"
                )

                model, hist_df, test_m, preds, budget = train_variant(
                    core=core, model=model, data=data, spec=spec, args=args,
                    device=device, run_dir=run_dir, run_tag=run_tag,
                )
                primary_scores = (
                    np.asarray(preds["pu_score"], dtype=float)
                    if spec.primary_output == "pu_score"
                    else np.asarray(preds["rank_score"], dtype=float)
                )
                ensemble_store.setdefault((scheme, spec.name), []).append(primary_scores.copy())

                metric_row = {
                    "split_scheme": scheme,
                    "repeat": int(rep),
                    "split_seed": split_seed,
                    "train_seed": train_seed,
                    "variant": spec.name,
                    "classification_objective": spec.cls_objective,
                    "ranking_objective": bool(spec.rank_objective),
                    "rank_sampler": spec.rank_sampler,
                    "primary_output": spec.primary_output,
                    "working_pi": float(args.pi),
                    **test_m,
                }
                all_metric_rows.append(metric_row)

                budget_rows.append({
                    "split_scheme": scheme,
                    "repeat": int(rep),
                    "split_seed": split_seed,
                    "train_seed": train_seed,
                    "variant": spec.name,
                    "primary_output": spec.primary_output,
                    "n_train_pos": n_train_pos,
                    "n_train_unlabeled": n_train_u,
                    "budget_mode": args.budget_mode,
                    **budget,
                })
                run_rows.append({
                    "split_scheme": scheme,
                    "repeat": int(rep),
                    "split_seed": split_seed,
                    "train_seed": train_seed,
                    "variant": spec.name,
                    "run_dir": str(run_dir),
                    "reviewer_role": spec.reviewer_role,
                })

                if args.save_per_repeat_rankings:
                    known = y == 1
                    rank_df = pd.DataFrame({
                        "gene_id": bundle["nodes"],
                        "primary_score": primary_scores,
                        "pu_score_Z": np.asarray(preds["pu_score"], dtype=float),
                        "rank_score_R": np.asarray(preds["rank_score"], dtype=float),
                        "final_Z_mc_uncertainty": np.asarray(preds["uncertainty"], dtype=float),
                        "rank_head_mc_uncertainty": np.asarray(preds["rank_head_std"], dtype=float),
                        "is_known_seed": known,
                    })
                    rank_df = rank_df.loc[~rank_df["is_known_seed"]].sort_values(
                        "primary_score", ascending=False
                    ).reset_index(drop=True)
                    rank_df.insert(0, "rank", np.arange(1, len(rank_df) + 1))
                    rank_df.to_csv(run_dir / f"{run_tag}_candidate_ranking.csv.gz", index=False, compression="gzip")

                # Free GPU memory between variants without changing the paired seed logic.
                del model, data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    per_split = pd.DataFrame(all_metric_rows)
    per_split.to_csv(save_root / "per_split_objective_controls.csv", index=False)
    pd.DataFrame(budget_rows).to_csv(save_root / "training_budget_audit.csv", index=False)
    pd.DataFrame(prior_rows).to_csv(save_root / "prior_and_label_audit.csv", index=False)
    pd.DataFrame(run_rows).to_csv(save_root / "run_index.csv", index=False)

    summary = summarize_by_variant(core, per_split, args)
    summary.to_csv(save_root / "summary_objective_controls_95CI.csv", index=False)
    reviewer_table = make_primary_reviewer_table(summary)
    reviewer_table.to_csv(save_root / "reviewer_primary_metrics_table.csv", index=False)

    contrasts = paired_contrasts(core, per_split, args)
    contrasts.to_csv(save_root / "paired_objective_contrasts.csv", index=False)
    primary_contrasts = make_primary_contrast_table(contrasts)
    primary_contrasts.to_csv(save_root / "reviewer_primary_paired_contrasts.csv", index=False)

    topks = sorted({int(x.strip()) for x in args.topk_overlap.split(",") if x.strip()})
    overlap = save_ensemble_rankings(ensemble_store, bundle, save_root, topks)
    overlap.to_csv(save_root / "ensemble_topk_overlap_with_full.csv", index=False)

    # Machine-readable short inventory of exactly which output answers which reviewer request.
    output_map = {
        "reviewer_primary_metrics_table.csv": "Mean/SD/95% CI for full, BCE, ranking-only, nnPU-only, and uniform-sampling controls.",
        "reviewer_primary_paired_contrasts.csv": "Paired within-split differences for the four prespecified reviewer-facing contrasts.",
        "training_budget_audit.csv": "Documents identical optimizer-step budgets, seeds, and checkpoint locations.",
        "prior_and_label_audit.csv": "Separates observed seed frequency from the working pi hyperparameter and records P/U counts.",
        "per_split_objective_controls.csv": "All held-out metrics for every repeat and variant.",
        "summary_objective_controls_95CI.csv": "Full metric summaries with bootstrap/t-based intervals from the reused main-code utility.",
        "ensemble_topk_overlap_with_full.csv": "Shows how objective/sampling choices change top-K candidate membership.",
        "ensemble_rankings/": "Genome-wide unlabeled-candidate ensemble rankings for every control variant.",
    }
    with open(save_root / "OUTPUTS_FOR_REVIEWER.json", "w", encoding="utf-8") as f:
        json.dump(output_map, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("[+] Objective-control experiment finished")
    print(f"[+] Main reviewer table: {save_root / 'reviewer_primary_metrics_table.csv'}")
    print(f"[+] Paired contrasts:    {save_root / 'reviewer_primary_paired_contrasts.csv'}")
    print(f"[+] Budget audit:        {save_root / 'training_budget_audit.csv'}")
    print(f"[+] Prior/label audit:   {save_root / 'prior_and_label_audit.csv'}")
    print(f"[+] Ensemble rankings:   {save_root / 'ensemble_rankings'}")
    print("=" * 100)


if __name__ == "__main__":
    main()

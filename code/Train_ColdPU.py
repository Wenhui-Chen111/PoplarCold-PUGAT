# -*- coding: utf-8 -*-
"""
Train_ColdPU.py
"""

import argparse
import copy
import json
import os
import sys
import platform
import random
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from scipy.stats import wilcoxon, t as student_t
    SCIPY_AVAILABLE = True
except Exception:
    wilcoxon = None
    student_t = None
    SCIPY_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Tee: dual stdout
# =========================================================
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, s):
        for f in self.files:
            try:
                f.write(s); f.flush()
            except Exception:
                pass
    def flush(self):
        for f in self.files:
            try: f.flush()
            except Exception: pass


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_run_name(prefix: str = "poplarcold_pugat") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def read_gene_ids(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def norm_gene_id(x) -> Optional[str]:
    """Normalize Potri IDs exactly as in Prepare_data.py."""
    if pd.isna(x):
        return None
    s = str(x).strip().split()[0]
    if s == "" or s.lower() == "nan":
        return None
    if s.startswith("Potri."):
        return re.sub(r"v\d+$", "", s)
    if s.startswith("POPTR_"):
        body = re.sub(r"v\d+$", "", s[len("POPTR_"):])
        return "Potri." + body
    if s.startswith("POPTR."):
        body = re.sub(r"v\d+$", "", s[len("POPTR."):])
        return "Potri." + body
    if s.startswith("Potri_"):
        body = re.sub(r"v\d+$", "", s[len("Potri_"):])
        return "Potri." + body
    return None


def load_seed_metadata(path: str,
                       sheet_name: Optional[str] = None,
                       gene_col: str = "Pt_Gene_ID",
                       group_col: str = "Grouping_Key",
                       tier_col: str = "Evidence_Tier") -> pd.DataFrame:
    """
    Load the audited seed metadata used only for grouped splitting and evidence-tier reporting.

    IMPORTANT: y.npy produced by Prepare_data.py is authoritative for labels.
    This function never overwrites the preprocessed labels.
    """
    if not path:
        raise ValueError("seed metadata path is empty")
    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name if sheet_name else 0)
    elif suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix in {".tsv", ".txt"} else ","
        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported seed metadata format: {suffix}")

    df = df.copy()

    # Reviewer-facing convenience: if the audited table has At_Gene_ID but no explicit
    # Grouping_Key, derive the grouping key from the Arabidopsis ortholog ID. Genes
    # without an At_Gene_ID (if any) become singleton groups so unrelated genes are
    # never collapsed into one empty group. The choice is printed and preserved in
    # the generated split manifest for auditability.
    if group_col not in df.columns and group_col == "Grouping_Key" and "At_Gene_ID" in df.columns:
        at = df["At_Gene_ID"].fillna("").astype(str).str.strip()
        gene_raw = df[gene_col].fillna("").astype(str).str.strip() if gene_col in df.columns else pd.Series([""] * len(df))
        df[group_col] = np.where(at.ne(""), at, "SELF::" + gene_raw)
        n_singletons = int(at.eq("").sum())
        print(
            f"[i] '{group_col}' not found; derived it from At_Gene_ID "
            f"({n_singletons} genes without At_Gene_ID assigned singleton groups)."
        )

    required = [gene_col, group_col, tier_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Seed metadata is missing required column(s): {missing}")

    df[f"{gene_col}_raw"] = df[gene_col]
    df[gene_col] = df[gene_col].apply(norm_gene_id)
    df[group_col] = df[group_col].fillna("").astype(str).str.strip()
    df[tier_col] = df[tier_col].fillna("").astype(str).str.strip()
    df = df[df[gene_col].notna()].copy()

    if df[gene_col].duplicated().any():
        dups = df.loc[df[gene_col].duplicated(keep=False), gene_col].tolist()
        raise ValueError(
            "Duplicate seed gene IDs after Potri-ID normalization "
            f"(first examples): {dups[:10]}"
        )
    return df


def collapse_evidence_tier(tier: str) -> str:
    """Collapse audited evidence tiers into A/B/B-C/C/D/E for reporting/splitting.

    Supports both verbose labels (e.g. ``Tier B ...``) and compact audited labels
    used in the revised spreadsheet (e.g. A2, B1, B2, B3, C, D1-D4, E2).
    """
    t = str(tier or "").strip()
    if not t or t.lower() == "nan":
        return "Unspecified"
    u = t.upper().replace("–", "-").replace("—", "-")
    if u.startswith("TIER "):
        u = u[5:].strip()
    if u.startswith("B/C") or u.startswith("B-C"):
        return "B/C"
    if u.startswith("A"):
        return "A"
    if u.startswith("B"):
        return "B"
    if u.startswith("C"):
        return "C"
    if u.startswith("D"):
        return "D"
    if u.startswith("E"):
        return "E"
    return "Unspecified"


def attach_seed_metadata(bundle: Dict,
                         metadata: pd.DataFrame,
                         gene_col: str = "Pt_Gene_ID",
                         group_col: str = "Grouping_Key",
                         tier_col: str = "Evidence_Tier") -> Dict:
    """
    Attach grouping/evidence metadata to graph nodes without altering y.npy.

    The positive labels are taken exclusively from the preprocessing output. Metadata is
    used to define homolog groups and evidence strata, and to audit label/metadata coverage.
    """
    out = dict(bundle)
    nodes = [norm_gene_id(g) or str(g).strip() for g in out["nodes"]]
    if len(set(nodes)) != len(nodes):
        raise ValueError("Duplicate graph gene IDs after normalization")
    out["nodes"] = nodes
    node_to_i = {g: i for i, g in enumerate(nodes)}
    meta_by_gene = metadata.set_index(gene_col, drop=False)

    node_group = np.array(["" for _ in nodes], dtype=object)
    node_tier = np.array(["" for _ in nodes], dtype=object)
    node_tier_group = np.array(["" for _ in nodes], dtype=object)

    metadata_genes = set(metadata[gene_col].dropna().astype(str).tolist())
    graph_genes = set(nodes)
    y = np.asarray(out["y"], dtype=np.int64)
    y_positive_genes = set(np.asarray(nodes, dtype=object)[y == 1].tolist())

    matched_genes = sorted(metadata_genes & graph_genes)
    missing_from_graph = sorted(metadata_genes - graph_genes)
    positive_with_metadata = sorted(y_positive_genes & metadata_genes)
    positive_missing_metadata = sorted(y_positive_genes - metadata_genes)
    metadata_graph_not_positive = sorted((metadata_genes & graph_genes) - y_positive_genes)

    for gene in matched_genes:
        i = node_to_i[gene]
        row = meta_by_gene.loc[gene]
        node_group[i] = str(row[group_col]).strip()
        node_tier[i] = str(row[tier_col]).strip()
        node_tier_group[i] = collapse_evidence_tier(row[tier_col])

    positive_missing_group = sorted([
        gene for gene in positive_with_metadata
        if str(meta_by_gene.loc[gene, group_col]).strip() == ""
    ])

    out["node_group_key"] = node_group
    out["node_evidence_tier"] = node_tier
    out["node_tier_group"] = node_tier_group
    out["seed_metadata_genes"] = metadata_genes
    out["seed_metadata_missing_from_graph"] = missing_from_graph
    out["seed_positive_missing_metadata"] = positive_missing_metadata
    out["seed_positive_missing_group"] = positive_missing_group
    out["seed_metadata_graph_not_positive"] = metadata_graph_not_positive
    out["seed_metadata_match_counts"] = {
        "metadata_unique_genes": int(len(metadata_genes)),
        "preprocessed_graph_nodes": int(len(nodes)),
        "preprocessed_y_positive_nodes": int(len(y_positive_genes)),
        "metadata_matched_to_graph": int(len(matched_genes)),
        "metadata_missing_from_graph": int(len(missing_from_graph)),
        "preprocessed_positive_with_metadata": int(len(positive_with_metadata)),
        "preprocessed_positive_missing_metadata": int(len(positive_missing_metadata)),
        "metadata_in_graph_not_positive": int(len(metadata_graph_not_positive)),
        "preprocessed_positive_missing_group_key": int(len(positive_missing_group)),
    }

    print(
        f"[i] Seed metadata audit: metadata={len(metadata_genes)}, "
        f"matched_to_graph={len(matched_genes)}, missing_from_graph={len(missing_from_graph)}, "
        f"y_positive={len(y_positive_genes)}, positive_with_metadata={len(positive_with_metadata)}, "
        f"positive_missing_metadata={len(positive_missing_metadata)}"
    )
    return out


def _random_positive_assignment(pos_genes: List[str], train_ratio: float,
                                val_ratio: float, seed: int) -> Dict[str, str]:
    """Legacy-compatible gene-level random split.

    The split seed controls ONLY the positive-set permutation via a local
    torch.Generator. It does not alter the global training RNG state.
    """
    genes = list(pos_genes)
    n = len(genes)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    n_train = max(1, int(n * train_ratio))
    n_val = int(n * val_ratio)
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)

    assignment: Dict[str, str] = {}
    for i in perm[:n_train]:
        assignment[str(genes[i])] = "train"
    for i in perm[n_train:n_train + n_val]:
        assignment[str(genes[i])] = "val"
    for i in perm[n_train + n_val:]:
        assignment[str(genes[i])] = "test"
    return assignment


def _partition_ratios(train_ratio: float, val_ratio: float) -> Dict[str, float]:
    """Return validated train/val/test ratios for positive-set splitting."""
    test_ratio = 1.0 - float(train_ratio) - float(val_ratio)
    ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }
    if any(v <= 0 for v in ratios.values()):
        raise ValueError(
            "train_ratio, val_ratio and test_ratio must all be > 0; "
            f"got {ratios}"
        )
    return ratios


def _largest_remainder_partition_counts(total: int,
                                        ratios: Dict[str, float]) -> Dict[str, int]:
    """Convert fractional targets into integer counts while preserving the total.

    The random-search grouped splitter uses these counts to keep the *number of
    homolog/orthology groups* in train/val/test as close as possible to the
    requested ratios. This is independent of model performance.
    """
    parts = ("train", "val", "test")
    if total < len(parts):
        raise ValueError(
            f"Need at least {len(parts)} groups for a 3-way grouped split; got {total}."
        )

    raw = {p: total * float(ratios[p]) for p in parts}
    counts = {p: int(np.floor(raw[p])) for p in parts}
    remainder = total - sum(counts.values())

    # Largest-remainder method; the part order provides deterministic tie-breaking.
    order = sorted(parts, key=lambda p: (raw[p] - counts[p], ratios[p]), reverse=True)
    for p in order[:remainder]:
        counts[p] += 1

    # With positive ratios and realistic group counts this is normally unnecessary,
    # but keep every partition non-empty for small group sets.
    for p in parts:
        if counts[p] == 0:
            donors = [q for q in parts if counts[q] > 1]
            if not donors:
                raise ValueError(f"Cannot make all grouped partitions non-empty: {counts}")
            donor = max(donors, key=lambda q: counts[q])
            counts[donor] -= 1
            counts[p] += 1

    if sum(counts.values()) != total:
        raise RuntimeError("Internal error while constructing integer group targets")
    return counts


def _build_group_metadata(pos_genes: List[str],
                          gene_to_group: Dict[str, str],
                          gene_to_tier: Optional[Dict[str, str]] = None):
    """Build group membership, group sizes and collapsed evidence-tier counts."""
    group_to_genes: Dict[str, List[str]] = {}
    for gene in pos_genes:
        group = str(gene_to_group.get(gene, "")).strip()
        if not group:
            raise ValueError(
                f"Positive gene {gene} has no grouping key. Fix the audited metadata "
                "or use --split_mode random."
            )
        group_to_genes.setdefault(group, []).append(gene)

    groups = sorted(group_to_genes.keys())
    gene_to_tier = gene_to_tier or {}
    group_to_tier_counts: Dict[str, Dict[str, int]] = {}
    tier_totals: Dict[str, int] = {}
    for group, genes in group_to_genes.items():
        d: Dict[str, int] = {}
        for gene in genes:
            tier = collapse_evidence_tier(gene_to_tier.get(gene, ""))
            d[tier] = d.get(tier, 0) + 1
            tier_totals[tier] = tier_totals.get(tier, 0) + 1
        group_to_tier_counts[group] = d

    tiers = sorted(tier_totals.keys())
    return group_to_genes, groups, group_to_tier_counts, tier_totals, tiers


def _score_group_assignment(group_assignment: Dict[str, str],
                            group_to_genes: Dict[str, List[str]],
                            group_to_tier_counts: Dict[str, Dict[str, int]],
                            tier_totals: Dict[str, int],
                            ratios: Dict[str, float],
                            gene_balance_weight: float = 1.0,
                            tier_balance_weight: float = 1.0) -> Dict[str, float]:
    """Score a grouped split using only pre-specified data-balance criteria.

    Lower is better. The score NEVER uses model predictions or test performance.

    Components:
      * gene_balance_loss: positive-gene counts vs requested train/val/test ratios
      * group_balance_loss: homolog-group counts vs requested ratios (diagnostic;
        random_search fixes integer group counts by construction)
      * tier_balance_loss: collapsed evidence-tier counts vs requested ratios,
        weighted by tier prevalence so very rare tiers do not dominate the split
    """
    parts = ("train", "val", "test")
    n_genes = float(sum(len(v) for v in group_to_genes.values()))
    n_groups = float(len(group_to_genes))

    gene_counts = {p: 0.0 for p in parts}
    group_counts = {p: 0.0 for p in parts}
    tier_counts = {t: {p: 0.0 for p in parts} for t in tier_totals}

    for group, part in group_assignment.items():
        if part not in parts:
            raise ValueError(f"Invalid grouped partition {part!r} for group {group!r}")
        group_counts[part] += 1.0
        gene_counts[part] += float(len(group_to_genes[group]))
        for tier, c in group_to_tier_counts[group].items():
            tier_counts[tier][part] += float(c)

    def rel_mse(actual: Dict[str, float], total: float) -> float:
        vals = []
        for p in parts:
            target = total * ratios[p]
            vals.append(((actual[p] - target) / max(target, 1.0)) ** 2)
        return float(np.mean(vals))

    gene_loss = rel_mse(gene_counts, n_genes)
    group_loss = rel_mse(group_counts, n_groups)

    tier_losses = []
    tier_weights = []
    for tier, total_i in tier_totals.items():
        if total_i <= 0:
            continue
        tier_losses.append(rel_mse(tier_counts[tier], float(total_i)))
        # Prevalence weighting prevents a one-gene tier from dominating the objective.
        tier_weights.append(float(total_i) / max(n_genes, 1.0))
    if tier_losses:
        tier_loss = float(np.average(tier_losses, weights=tier_weights))
    else:
        tier_loss = 0.0

    total_score = (
        float(gene_balance_weight) * gene_loss
        + float(tier_balance_weight) * tier_loss
    )
    return {
        "objective": float(total_score),
        "gene_balance_loss": float(gene_loss),
        "group_balance_loss": float(group_loss),
        "tier_balance_loss": float(tier_loss),
    }


def _legacy_grouped_positive_assignment(pos_genes: List[str],
                                        gene_to_group: Dict[str, str],
                                        train_ratio: float, val_ratio: float,
                                        seed: int) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Original greedy grouped splitter retained for reproducibility/comparison."""
    group_to_genes, groups, group_to_tier_counts, tier_totals, _ = _build_group_metadata(
        pos_genes, gene_to_group, {}
    )
    ratios = _partition_ratios(train_ratio, val_ratio)
    targets = {p: len(pos_genes) * ratios[p] for p in ratios}

    rng = np.random.default_rng(seed)
    groups = list(groups)
    rng.shuffle(groups)
    groups.sort(key=lambda g: len(group_to_genes[g]), reverse=True)

    counts = {p: 0 for p in ratios}
    group_assignment: Dict[str, str] = {}
    for group in groups:
        size = len(group_to_genes[group])
        candidates = []
        for part in ("train", "val", "test"):
            new_counts = dict(counts)
            new_counts[part] += size
            score = sum(
                ((new_counts[p] - targets[p]) / max(targets[p], 1.0)) ** 2
                for p in targets
            )
            candidates.append((score, float(rng.random()) * 1e-12, part))
        _, _, chosen = min(candidates)
        group_assignment[group] = chosen
        counts[chosen] += size

    assignment: Dict[str, str] = {}
    for group, genes in group_to_genes.items():
        for gene in genes:
            assignment[gene] = group_assignment[group]

    score = _score_group_assignment(
        group_assignment, group_to_genes, group_to_tier_counts, tier_totals,
        ratios, gene_balance_weight=1.0, tier_balance_weight=0.0
    )
    return assignment, score


def _random_search_grouped_positive_assignment(
        pos_genes: List[str],
        gene_to_group: Dict[str, str],
        gene_to_tier: Dict[str, str],
        train_ratio: float,
        val_ratio: float,
        seed: int,
        trials: int = 20000,
        gene_balance_weight: float = 1.0,
        tier_balance_weight: float = 1.0,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Balanced homolog-group-exclusive split selected without model outcomes.

    For every trial:
      1. Randomly permute whole homolog/orthology groups.
      2. Allocate an integer number of groups to train/val/test using the
         requested ratios (largest-remainder rule), so group counts are stable.
      3. Score the candidate by positive-gene balance and evidence-tier balance.

    The candidate with the smallest *pre-model* balance objective is returned.
    No prediction, validation metric, or test metric enters split selection, so this
    does not cherry-pick an "easy" split.
    """
    if int(trials) < 1:
        raise ValueError(f"group_split_trials must be >= 1; got {trials}")
    if gene_balance_weight < 0 or tier_balance_weight < 0:
        raise ValueError("group split balance weights must be non-negative")

    ratios = _partition_ratios(train_ratio, val_ratio)
    group_to_genes, groups, group_to_tier_counts, tier_totals, _ = _build_group_metadata(
        pos_genes, gene_to_group, gene_to_tier
    )
    group_target_counts = _largest_remainder_partition_counts(len(groups), ratios)

    rng = np.random.default_rng(seed)
    groups_arr = np.asarray(groups, dtype=object)
    parts = ("train", "val", "test")

    best_assignment: Optional[Dict[str, str]] = None
    best_score: Optional[Dict[str, float]] = None

    for _trial in range(int(trials)):
        perm = rng.permutation(groups_arr)
        group_assignment: Dict[str, str] = {}
        cursor = 0
        for part in parts:
            take = int(group_target_counts[part])
            for group in perm[cursor:cursor + take]:
                group_assignment[str(group)] = part
            cursor += take

        score = _score_group_assignment(
            group_assignment=group_assignment,
            group_to_genes=group_to_genes,
            group_to_tier_counts=group_to_tier_counts,
            tier_totals=tier_totals,
            ratios=ratios,
            gene_balance_weight=gene_balance_weight,
            tier_balance_weight=tier_balance_weight,
        )
        # Deterministic first-best rule for a fixed RNG seed.
        if best_score is None or score["objective"] < best_score["objective"] - 1e-15:
            best_score = score
            best_assignment = dict(group_assignment)

    if best_assignment is None or best_score is None:
        raise RuntimeError("random_search grouped split failed to produce an assignment")

    assignment: Dict[str, str] = {}
    for group, genes in group_to_genes.items():
        part = best_assignment[group]
        for gene in genes:
            assignment[gene] = part

    for part in parts:
        if sum(v == part for v in assignment.values()) == 0:
            raise RuntimeError(f"Grouped split produced an empty {part} partition")

    best_score = dict(best_score)
    best_score.update({
        "n_groups_total": int(len(groups)),
        "target_train_groups": int(group_target_counts["train"]),
        "target_val_groups": int(group_target_counts["val"]),
        "target_test_groups": int(group_target_counts["test"]),
        "group_split_trials": int(trials),
    })
    return assignment, best_score


def _grouped_positive_assignment(
        pos_genes: List[str],
        gene_to_group: Dict[str, str],
        gene_to_tier: Dict[str, str],
        train_ratio: float,
        val_ratio: float,
        seed: int,
        strategy: str = "random_search",
        trials: int = 20000,
        gene_balance_weight: float = 1.0,
        tier_balance_weight: float = 1.0,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Dispatch grouped splitting strategy."""
    if strategy == "legacy_greedy":
        return _legacy_grouped_positive_assignment(
            pos_genes, gene_to_group, train_ratio, val_ratio, seed
        )
    if strategy == "random_search":
        return _random_search_grouped_positive_assignment(
            pos_genes=pos_genes,
            gene_to_group=gene_to_group,
            gene_to_tier=gene_to_tier,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            trials=trials,
            gene_balance_weight=gene_balance_weight,
            tier_balance_weight=tier_balance_weight,
        )
    raise ValueError(
        f"Unknown group_split_strategy={strategy!r}; "
        "choose from {'legacy_greedy', 'random_search'}"
    )


def generate_split_manifest(bundle: Dict, metadata: Optional[pd.DataFrame],
                            split_mode: str, num_repeats: int,
                            seed: int, train_ratio: float, val_ratio: float,
                            gene_col: str = "Pt_Gene_ID",
                            group_col: str = "Grouping_Key",
                            tier_col: str = "Evidence_Tier",
                            group_split_strategy: str = "random_search",
                            group_split_trials: int = 20000,
                            group_gene_balance_weight: float = 1.0,
                            group_tier_balance_weight: float = 1.0) -> pd.DataFrame:
    nodes = list(bundle["nodes"])
    y = np.asarray(bundle["y"], dtype=np.int64)
    pos_genes = [nodes[i] for i in np.where(y == 1)[0]]
    if len(pos_genes) < 5:
        raise ValueError(f"Too few positives after metadata/filtering: {len(pos_genes)}")

    gene_to_group: Dict[str, str] = {}
    gene_to_tier: Dict[str, str] = {}
    if metadata is not None:
        gene_to_group = dict(zip(metadata[gene_col], metadata[group_col]))
        gene_to_tier = dict(zip(metadata[gene_col], metadata[tier_col]))

    if split_mode == "grouped" and metadata is None:
        raise ValueError("--split_mode grouped requires --seed_metadata")

    rows = []
    for r in range(num_repeats):
        split_seed = int(seed + r)
        split_diag: Dict[str, float] = {}
        if split_mode == "grouped":
            assignment, split_diag = _grouped_positive_assignment(
                pos_genes=pos_genes,
                gene_to_group=gene_to_group,
                gene_to_tier=gene_to_tier,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=split_seed,
                strategy=group_split_strategy,
                trials=group_split_trials,
                gene_balance_weight=group_gene_balance_weight,
                tier_balance_weight=group_tier_balance_weight,
            )
        elif split_mode == "random":
            assignment = _random_positive_assignment(
                pos_genes, train_ratio, val_ratio, split_seed
            )
            split_diag = {
                "objective": np.nan,
                "gene_balance_loss": np.nan,
                "group_balance_loss": np.nan,
                "tier_balance_loss": np.nan,
            }
        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

        for gene in pos_genes:
            tier = str(gene_to_tier.get(gene, "")).strip()
            rows.append({
                "repeat": r + 1,
                "split_seed": split_seed,
                "gene_id": gene,
                "group_key": str(gene_to_group.get(gene, "")).strip(),
                "evidence_tier": tier,
                "tier_group": collapse_evidence_tier(tier),
                "partition": assignment[gene],
                "split_strategy": group_split_strategy if split_mode == "grouped" else "gene_random",
                "split_objective": split_diag.get("objective", np.nan),
                "gene_balance_loss": split_diag.get("gene_balance_loss", np.nan),
                "group_balance_loss": split_diag.get("group_balance_loss", np.nan),
                "tier_balance_loss": split_diag.get("tier_balance_loss", np.nan),
            })

    manifest = pd.DataFrame(rows)
    validate_split_manifest(manifest, pos_genes, split_mode=split_mode)
    return manifest


def save_split_balance_reports(manifest: pd.DataFrame,
                               save_root: str,
                               train_ratio: float,
                               val_ratio: float) -> None:
    """Save auditable gene/group/evidence-tier balance reports for every repeat."""
    parts = ("train", "val", "test")
    ratios = _partition_ratios(train_ratio, val_ratio)

    # 1) Positive-gene counts (same information as the historical file, explicit targets added).
    gene_counts = (
        manifest.groupby(["repeat", "partition"]).size()
        .unstack(fill_value=0)
        .reindex(columns=list(parts), fill_value=0)
    )
    gene_counts.to_csv(os.path.join(save_root, "split_partition_counts.csv"))

    # 2) Unique homolog/orthology group counts.
    if "group_key" in manifest.columns and manifest["group_key"].astype(str).str.strip().ne("").any():
        group_counts = (
            manifest[manifest["group_key"].astype(str).str.strip().ne("")]
            .groupby(["repeat", "partition"])["group_key"].nunique()
            .unstack(fill_value=0)
            .reindex(columns=list(parts), fill_value=0)
        )
        group_counts.to_csv(os.path.join(save_root, "split_group_counts.csv"))
    else:
        group_counts = pd.DataFrame(index=gene_counts.index, columns=list(parts), data=np.nan)

    # 3) Long-form evidence-tier counts by repeat/partition.
    if "tier_group" in manifest.columns:
        tier_counts = (
            manifest.groupby(["repeat", "partition", "tier_group"]).size()
            .rename("n_positive_genes").reset_index()
        )
        tier_counts.to_csv(os.path.join(save_root, "split_evidence_tier_counts.csv"), index=False)

    # 4) One row per repeat/partition with observed and target genes/groups.
    detail_rows = []
    summary_rows = []
    for rep, sub in manifest.groupby("repeat"):
        n_gene_total = int(len(sub))
        nonempty_groups = sub.loc[sub["group_key"].astype(str).str.strip().ne(""), "group_key"] \
            if "group_key" in sub.columns else pd.Series(dtype=object)
        n_group_total = int(nonempty_groups.nunique()) if len(nonempty_groups) else 0

        for part in parts:
            obs_genes = int((sub["partition"] == part).sum())
            obs_groups = (
                int(sub.loc[sub["partition"] == part, "group_key"].astype(str)
                    .loc[lambda s: s.str.strip().ne("")].nunique())
                if "group_key" in sub.columns else 0
            )
            detail_rows.append({
                "repeat": int(rep),
                "split_seed": int(sub["split_seed"].iloc[0]),
                "partition": part,
                "n_positive_genes": obs_genes,
                "target_positive_genes": n_gene_total * ratios[part],
                "positive_gene_ratio": obs_genes / max(n_gene_total, 1),
                "n_groups": obs_groups if n_group_total else np.nan,
                "target_groups": n_group_total * ratios[part] if n_group_total else np.nan,
                "group_ratio": obs_groups / max(n_group_total, 1) if n_group_total else np.nan,
            })

        first = sub.iloc[0]
        summary_rows.append({
            "repeat": int(rep),
            "split_seed": int(first["split_seed"]),
            "split_strategy": first.get("split_strategy", ""),
            "split_objective": first.get("split_objective", np.nan),
            "gene_balance_loss": first.get("gene_balance_loss", np.nan),
            "group_balance_loss": first.get("group_balance_loss", np.nan),
            "tier_balance_loss": first.get("tier_balance_loss", np.nan),
            "n_positive_genes_total": n_gene_total,
            "n_groups_total": n_group_total if n_group_total else np.nan,
        })

    pd.DataFrame(detail_rows).to_csv(
        os.path.join(save_root, "split_balance_by_partition.csv"), index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(save_root, "split_balance_summary.csv"), index=False
    )

    print("[i] Split positive-gene counts (first 10 repeats):")
    print(gene_counts.head(10).to_string())
    if not group_counts.empty and group_counts.notna().any().any():
        print("[i] Split homolog-group counts (first 10 repeats):")
        print(group_counts.head(10).to_string())


def validate_split_manifest(manifest: pd.DataFrame, pos_genes: List[str],
                            split_mode: str = "grouped") -> None:
    required = {"repeat", "split_seed", "gene_id", "partition"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Split manifest missing columns: {sorted(missing)}")
    valid_parts = {"train", "val", "test"}
    if not set(manifest["partition"]).issubset(valid_parts):
        raise ValueError("Split manifest partition must be train/val/test")

    expected = set(pos_genes)
    for rep, sub in manifest.groupby("repeat"):
        observed = set(sub["gene_id"])
        if observed != expected:
            miss = sorted(expected - observed)[:10]
            extra = sorted(observed - expected)[:10]
            raise ValueError(
                f"Manifest repeat {rep} does not match current positives; missing={miss}, extra={extra}"
            )
        if sub["gene_id"].duplicated().any():
            raise ValueError(f"Duplicate genes in manifest repeat {rep}")
        for part in valid_parts:
            if int((sub["partition"] == part).sum()) == 0:
                raise ValueError(f"Manifest repeat {rep} has empty {part} partition")
        if split_mode == "grouped" and "group_key" in sub.columns:
            leak = sub.groupby("group_key")["partition"].nunique()
            leak = leak[(leak.index.astype(str) != "") & (leak > 1)]
            if len(leak):
                raise ValueError(
                    f"Group leakage in manifest repeat {rep}: {leak.index.tolist()[:10]}"
                )


def build_pu_masks_from_manifest(data: Data, nodes: List[str],
                                 manifest_repeat: pd.DataFrame,
                                 hidden_seed_as_unlabeled: bool = False) -> Data:
    assignment = dict(zip(manifest_repeat["gene_id"], manifest_repeat["partition"]))
    y = data.y
    train_pos_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_pos_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_pos_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    for i, gene in enumerate(nodes):
        if int(y[i].item()) != 1:
            continue
        part = assignment.get(gene)
        if part == "train":
            train_pos_mask[i] = True
        elif part == "val":
            val_pos_mask[i] = True
        elif part == "test":
            test_pos_mask[i] = True
        else:
            raise ValueError(f"Positive gene {gene} missing from split manifest")

    true_unlabeled_mask = (y == 0)
    hidden_seed_mask = val_pos_mask | test_pos_mask
    train_unlabeled_mask = true_unlabeled_mask.clone()
    if hidden_seed_as_unlabeled:
        # Robustness setting requested by reviewer: held-out seeds participate in the
        # PU unlabeled distribution and ranking-negative sampling exactly like unknown genes.
        train_unlabeled_mask = train_unlabeled_mask | hidden_seed_mask

    data.train_pos_mask = train_pos_mask
    data.val_pos_mask = val_pos_mask
    data.test_pos_mask = test_pos_mask
    data.unlabeled_mask = true_unlabeled_mask          # evaluation universe only
    data.train_unlabeled_mask = train_unlabeled_mask  # loss/ranking pool only
    data.hidden_seed_mask = hidden_seed_mask
    data.known_pos_mask = (y == 1)
    return data


# =========================================================
# Data loading
# =========================================================
def _load_edges(data_dir: Path, name_prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    edge_path = data_dir / f"edges_{name_prefix}.npy"
    weight_path = data_dir / f"edges_{name_prefix}_weight.npy"
    if not edge_path.exists() or not weight_path.exists():
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    edge_index = np.load(edge_path).astype(np.int64)
    edge_weight = np.load(weight_path).astype(np.float32)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"{edge_path} should have shape [2, E]")
    if edge_weight.shape[0] != edge_index.shape[1]:
        raise ValueError(f"{weight_path} length mismatch with edges")
    return edge_index, edge_weight


def combine_edges(coexp_e, coexp_w, ppi_e, ppi_w,
                  coexp_ratio=0.5, ppi_ratio=0.5,
                  use_coexp=True, use_ppi=True):
    """Merge coexp / ppi edges into a single graph (for non-hetero mode)."""
    a, b = float(coexp_ratio) if use_coexp else 0.0, float(ppi_ratio) if use_ppi else 0.0
    s = a + b
    if s <= 0:
        raise ValueError("At least one graph must be enabled")
    a, b = a / s, b / s
    parts = []
    if use_coexp and coexp_e.shape[1] > 0:
        parts.append((coexp_e, coexp_w * a))
    if use_ppi and ppi_e.shape[1] > 0:
        parts.append((ppi_e, ppi_w * b))
    if len(parts) == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    edge_index = np.concatenate([p[0] for p in parts], axis=1)
    edge_weight = np.concatenate([p[1] for p in parts], axis=0)
    df = pd.DataFrame({"src": edge_index[0], "dst": edge_index[1], "weight": edge_weight})
    df = df.groupby(["src", "dst"], as_index=False)["weight"].sum()
    df["weight"] = df["weight"].clip(lower=0.0, upper=1.0)
    merged_e = df[["src", "dst"]].to_numpy(dtype=np.int64).T
    merged_w = df["weight"].to_numpy(dtype=np.float32)
    return merged_e, merged_w



def load_preprocessed(data_dir: str):
    """
    Load preprocessed output:
      - X.npy / y.npy / gene_ids.txt
      - edges_coexp.npy / edges_coexp_weight.npy
      - edges_ppi.npy / edges_ppi_weight.npy
      - Optional cond_input.npy / other_input.npy
    """
    d = Path(data_dir)
    required = ["X.npy", "y.npy", "gene_ids.txt",
                "edges_coexp.npy", "edges_coexp_weight.npy",
                "edges_ppi.npy", "edges_ppi_weight.npy"]
    missing = [f for f in required if not (d / f).exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    X = np.load(d / "X.npy").astype(np.float32)
    y = np.load(d / "y.npy").astype(np.int64)
    nodes = read_gene_ids(d / "gene_ids.txt")

    fn_path = d / "feature_names.txt"
    feature_names = read_gene_ids(fn_path) if fn_path.exists() else [f"x_{i}" for i in range(X.shape[1])]

    coexp_e, coexp_w = _load_edges(d, "coexp")
    ppi_e, ppi_w = _load_edges(d, "ppi")

    cond_input = None
    other_input = None
    if (d / "cond_input.npy").exists():
        cond_input = np.load(d / "cond_input.npy").astype(np.float32)
    if (d / "other_input.npy").exists():
        other_input = np.load(d / "other_input.npy").astype(np.float32)

    print(f"[i] Loaded: X={X.shape}, y_pos={int((y==1).sum())}, "
          f"coexp_edges={coexp_e.shape[1]}, ppi_edges={ppi_e.shape[1]}")
    if cond_input is not None:
        print(f"[i] cond_input: {cond_input.shape}")
    if other_input is not None:
        print(f"[i] other_input: {other_input.shape}")

    return {
        "X": X, "y": y, "nodes": nodes, "feature_names": feature_names,
        "coexp_e": coexp_e, "coexp_w": coexp_w,
        "ppi_e": ppi_e, "ppi_w": ppi_w,
        "cond_input": cond_input, "other_input": other_input,
    }


def build_data_for_split(
    bundle: Dict,
    manifest_repeat: pd.DataFrame,
    hidden_seed_as_unlabeled: bool,
    use_coexp: bool,
    use_ppi: bool,
    coexp_ratio: float,
    ppi_ratio: float,
    use_hetero: bool,
) -> Data:
    """
    Build Data object for one split:
      1. Directly use saved PPI graph from preprocessing
      2. If use_hetero, store coexp and ppi separately in data.edge_index_coexp / data.edge_index_ppi
         else merge into single edge_index
    """
    X = bundle["X"]; y = bundle["y"]
    coexp_e, coexp_w = bundle["coexp_e"], bundle["coexp_w"]
    ppi_e, ppi_w = bundle["ppi_e"], bundle["ppi_w"]

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        num_nodes=X.shape[0],
    )

    # Attach cond/other features to data for model access
    if bundle["cond_input"] is not None:
        data.cond_input = torch.tensor(bundle["cond_input"], dtype=torch.float32)
    if bundle["other_input"] is not None:
        data.other_input = torch.tensor(bundle["other_input"], dtype=torch.float32)

    data = build_pu_masks_from_manifest(
        data, nodes=bundle["nodes"], manifest_repeat=manifest_repeat,
        hidden_seed_as_unlabeled=hidden_seed_as_unlabeled
    )

    # Use preprocessed PPI graph directly
    ppi_use_e, ppi_use_w = ppi_e, ppi_w

    if use_hetero:
        # Heterogeneous: separate branches
        ce, cw = (coexp_e, coexp_w) if use_coexp else (np.empty((2,0), np.int64), np.empty((0,), np.float32))
        pe, pw = (ppi_use_e, ppi_use_w) if use_ppi else (np.empty((2,0), np.int64), np.empty((0,), np.float32))
        # Add self-loop for empty graph to avoid GAT errors
        if ce.shape[1] == 0:
            n = data.num_nodes
            ce = np.vstack([np.arange(n), np.arange(n)]).astype(np.int64)
            cw = np.ones(n, dtype=np.float32) * 1e-4
        if pe.shape[1] == 0:
            n = data.num_nodes
            pe = np.vstack([np.arange(n), np.arange(n)]).astype(np.int64)
            pw = np.ones(n, dtype=np.float32) * 1e-4
        data.edge_index_coexp = torch.tensor(ce, dtype=torch.long)
        data.edge_weight_coexp = torch.tensor(cw, dtype=torch.float32)
        data.edge_index_ppi = torch.tensor(pe, dtype=torch.long)
        data.edge_weight_ppi = torch.tensor(pw, dtype=torch.float32)
        # Also attach merged for feature-only compatibility
        data.edge_index = data.edge_index_coexp
        data.edge_weight = data.edge_weight_coexp
    else:
        merged_e, merged_w = combine_edges(coexp_e, coexp_w, ppi_use_e, ppi_use_w,
                                           coexp_ratio=coexp_ratio, ppi_ratio=ppi_ratio,
                                           use_coexp=use_coexp, use_ppi=use_ppi)
        if merged_e.shape[1] == 0:
            n = data.num_nodes
            merged_e = np.vstack([np.arange(n), np.arange(n)]).astype(np.int64)
            merged_w = np.ones(n, dtype=np.float32) * 1e-4
        data.edge_index = torch.tensor(merged_e, dtype=torch.long)
        data.edge_weight = torch.tensor(merged_w, dtype=torch.float32)

    return data


# =========================================================
# Class prior estimation
# =========================================================
def estimate_pi(method: str,
                n_pos: int, n_nodes: int,
                pi_override: float = -1.0,
                X: np.ndarray = None,
                pos_mask: np.ndarray = None) -> float:
    """
    method:
      fixed: n_pos/n_nodes (traditional, often biased small)
      manual: use pi_override (recommend 0.015, 0.02, 0.03, 0.05)
      tice: simple threshold-based estimation, clip to [0.005, 0.05]
    """
    if method == "fixed":
        return float(n_pos) / float(n_nodes)
    if method == "manual":
        if pi_override <= 0:
            raise ValueError("--pi_method manual requires --pi_override > 0")
        return float(pi_override)
    if method == "tice":
        # Simple approximation: use PCA 1st component as score
        if X is None or pos_mask is None:
            return float(n_pos) / float(n_nodes)
        from sklearn.decomposition import PCA
        pc1 = PCA(n_components=1).fit_transform(X).reshape(-1)
        pos_scores = pc1[pos_mask]
        all_scores = pc1
        # Estimate positive density in high-score region
        thr = np.quantile(pos_scores, 0.5)
        n_above_all = float((all_scores >= thr).sum())
        n_above_pos = float((pos_scores >= thr).sum())
        if n_above_all <= 0:
            return float(n_pos) / float(n_nodes)
        est = (n_pos * n_above_all) / (float(n_nodes) * n_above_pos + 1e-8)
        return float(np.clip(est, 0.005, 0.05))
    raise ValueError(f"Unknown pi method: {method}")


# =========================================================
# Model: hetero-aware + cross-stress contrastive
# =========================================================
class SingleOmicPUGAT(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 proj_dim: int = 64,
                 heads: int = 4,
                 dropout: float = 0.3,
                 use_hetero: bool = False):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.dropout_p = dropout
        self.use_hetero = bool(use_hetero)

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        # Feature-only head
        self.feature_head = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Edge encoder (shared)
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16),
        )

        # GNN branches
        if use_hetero:
            self.gnn_coexp = GATv2Conv(proj_dim, hidden_dim // heads,
                                       heads=heads, concat=True, dropout=dropout, edge_dim=16)
            self.gnn_ppi = GATv2Conv(proj_dim, hidden_dim // heads,
                                     heads=heads, concat=True, dropout=dropout, edge_dim=16)
            self.bn_coexp = nn.BatchNorm1d(hidden_dim)
            self.bn_ppi = nn.BatchNorm1d(hidden_dim)
            # Gated fusion for two relations
            self.rel_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2), nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.gnn1 = GATv2Conv(proj_dim, hidden_dim // heads,
                                  heads=heads, concat=True, dropout=dropout, edge_dim=16)
            self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.rank_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.mix_gate = nn.Sequential(
            nn.Linear(proj_dim + hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )

    def _graph_branch(self, h0, data):
        if self.use_hetero:
            ew_c = data.edge_weight_coexp
            ea_c = self.edge_encoder(ew_c.view(-1, 1))
            h_c = self.gnn_coexp(h0, data.edge_index_coexp, edge_attr=ea_c)
            h_c = self.bn_coexp(h_c); h_c = F.relu(h_c)
            h_c = F.dropout(h_c, p=self.dropout_p, training=self.training)

            ew_p = data.edge_weight_ppi
            ea_p = self.edge_encoder(ew_p.view(-1, 1))
            h_p = self.gnn_ppi(h0, data.edge_index_ppi, edge_attr=ea_p)
            h_p = self.bn_ppi(h_p); h_p = F.relu(h_p)
            h_p = F.dropout(h_p, p=self.dropout_p, training=self.training)

            alpha = torch.sigmoid(self.rel_gate(torch.cat([h_c, h_p], dim=1)))
            h = alpha * h_c + (1.0 - alpha) * h_p

            return h
        else:
            ew = data.edge_weight
            ea = self.edge_encoder(ew.view(-1, 1))
            h = self.gnn1(h0, data.edge_index, edge_attr=ea)
            h = self.bn1(h); h = F.relu(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)

            return h

    def forward(self, data: Data, return_details: bool = False):
        h0 = self.feature_encoder(data.x)
        feature_logit = self.feature_head(h0).squeeze(-1)
        h_graph = self._graph_branch(h0, data)
        graph_logit = self.graph_head(h_graph).squeeze(-1)
        rank_logit = self.rank_head(h_graph).squeeze(-1)
        mix_alpha = torch.sigmoid(self.mix_gate(torch.cat([h0, h_graph], dim=1))).squeeze(-1)
        final_logit = mix_alpha * feature_logit + (1.0 - mix_alpha) * graph_logit

        out = {
            "feature_logit": feature_logit,
            "graph_logit": graph_logit,
            "final_logit": final_logit,
            "rank_logit": rank_logit,
            "mix_alpha": mix_alpha,
            "embedding": h_graph,
        }

        if return_details:
            return out
        return final_logit


# =========================================================
# Losses
# =========================================================
class PULossWithRanking(nn.Module):
    def __init__(self, pi, beta=0.0, rank_weight=1.0,
                 margin=0.3, num_rank_pairs=2048, hard_negative_ratio=0.5):
        super().__init__()
        self.pi = float(pi); self.beta = float(beta)
        self.rank_weight = float(rank_weight)
        self.margin = float(margin); self.num_rank_pairs = int(num_rank_pairs)
        self.hard_negative_ratio = float(hard_negative_ratio)

    def nnpu_part(self, logits, pos_mask, unlabeled_mask):
        l_pos = F.softplus(-logits); l_neg = F.softplus(logits)
        Rp_pos = l_pos[pos_mask].mean(); Rp_neg = l_neg[pos_mask].mean()
        eff_un = unlabeled_mask.clone()

        if eff_un.sum() > 0:
            Ru_neg = l_neg[eff_un].mean()
        else:
            Ru_neg = torch.tensor(0.0, device=logits.device)
        neg_risk_raw = Ru_neg - self.pi * Rp_neg
        neg_risk_used = (torch.clamp(neg_risk_raw, min=0.0) if self.beta == 0.0
                         else torch.clamp(neg_risk_raw, min=-self.beta))
        pu_loss = self.pi * Rp_pos + neg_risk_used

        stats = {"Rp_pos": Rp_pos.detach(), "Rp_neg": Rp_neg.detach(),
                 "Ru_neg": Ru_neg.detach(), "neg_risk_raw": neg_risk_raw.detach(),
                 "neg_risk_used": neg_risk_used.detach(),
                 "pi": torch.tensor(self.pi, device=logits.device)}
        return pu_loss, stats

    def ranking_part(self, rank_logits, pos_mask, unlabeled_mask):
        pos_idx = torch.where(pos_mask)[0]
        neg_mask = unlabeled_mask.clone()
        
        neg_idx = torch.where(neg_mask)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return torch.tensor(0.0, device=rank_logits.device)
        pos_sample = pos_idx[torch.randint(0, len(pos_idx), (self.num_rank_pairs,),
                                           device=rank_logits.device)]
        neg_scores = rank_logits[neg_idx].detach()
        hard_k = max(1, min(int(len(neg_idx) * self.hard_negative_ratio), len(neg_idx)))
        _, order = torch.topk(neg_scores, k=hard_k, largest=True)
        hard_neg = neg_idx[order]
        neg_pool = hard_neg if len(hard_neg) > 0 else neg_idx
        neg_sample = neg_pool[torch.randint(0, len(neg_pool), (self.num_rank_pairs,),
                                            device=rank_logits.device)]
        return F.relu(self.margin - (rank_logits[pos_sample] - rank_logits[neg_sample])).mean()

    def forward(self, final_logit, rank_logit, pos_mask, unlabeled_mask):
        pu_loss, stats = self.nnpu_part(final_logit, pos_mask, unlabeled_mask)
        rank_loss = self.ranking_part(rank_logit, pos_mask, unlabeled_mask)
        total = pu_loss + self.rank_weight * rank_loss
        stats["rank_loss"] = rank_loss.detach()
        return total, stats



# =========================================================
# Metrics / prediction
# =========================================================
def compute_ranking_metrics(scores, eval_pos_mask, unlabeled_mask, Ks=(50, 100, 200)):
    """
    Compute ranking metrics on one explicit evaluation universe:

        candidate set = held-out positives for this split + truly unlabeled genes

    Validation: eval_pos_mask = val_pos_mask
                candidate set = val positives + unlabeled genes

    Testing:    eval_pos_mask = test_pos_mask
                candidate set = test positives + unlabeled genes

    Positives from the other split are therefore excluded from the metric pool,
    instead of being incorrectly counted as y=0 background samples.
    """
    eval_pos = eval_pos_mask.detach().cpu().numpy().astype(bool)
    unlabeled = unlabeled_mask.detach().cpu().numpy().astype(bool)

    if eval_pos.shape != unlabeled.shape:
        raise ValueError("eval_pos_mask and unlabeled_mask must have the same shape")
    if np.any(eval_pos & unlabeled):
        raise ValueError("eval_pos_mask and unlabeled_mask must be disjoint")

    # Correct evaluation universe: current held-out positives + true unlabeled genes.
    candidate_mask = eval_pos | unlabeled
    cand = np.where(candidate_mask)[0]

    if len(cand) == 0:
        out = {"MRR": 0.0, "AUPRC": 0.0, "AUROC": 0.0}
        for K in Ks:
            out[f"Hits@{K}"] = 0
            out[f"Precision@{K}"] = 0.0
            out[f"Recall@{K}"] = 0.0
            out[f"nDCG@{K}"] = 0.0
        return out

    cs = scores[cand]
    order = np.argsort(-cs)
    ranked = cand[order]

    hidden = set(np.where(eval_pos)[0].tolist())
    hits = [1 if n in hidden else 0 for n in ranked]

    mrr = 0.0
    for i, h in enumerate(hits, start=1):
        if h == 1:
            mrr = 1.0 / i
            break

    def dcg(hk):
        return sum(h / np.log2(i + 1) for i, h in enumerate(hk, start=1))

    m = {"MRR": float(mrr)}
    denom = max(1, len(hidden))
    for K in Ks:
        topk = hits[:K]
        hit_count = int(sum(topk))
        idcg_ideal = [1] * min(len(hidden), K) + [0] * max(0, K - min(len(hidden), K))
        idcg = dcg(idcg_ideal) if len(hidden) > 0 else 0.0
        ndcg = (dcg(topk) / idcg) if idcg > 0 else 0.0
        m[f"Hits@{K}"] = hit_count
        m[f"Precision@{K}"] = hit_count / K
        m[f"Recall@{K}"] = hit_count / denom
        m[f"nDCG@{K}"] = float(ndcg)

    from sklearn.metrics import average_precision_score, roc_auc_score

    # Within candidate_mask, only this split's held-out positives are y=1;
    # all remaining candidates are genuine unlabeled genes.
    y_true = eval_pos[cand].astype(int)
    if y_true.sum() > 0:
        m["AUPRC"] = float(average_precision_score(y_true, cs))
    else:
        m["AUPRC"] = 0.0

    if 0 < y_true.sum() < len(y_true):
        m["AUROC"] = float(roc_auc_score(y_true, cs))
    else:
        m["AUROC"] = 0.0

    return m


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model, data, mc_times=30, uncertainty_penalty=0.10):
    model.eval()
    feats, graphs, finals, ranks, mixes = [], [], [], [], []
    for _ in range(mc_times):
        enable_dropout(model)
        out = model(data, return_details=True)
        feats.append(torch.sigmoid(out["feature_logit"]).detach().cpu())
        graphs.append(torch.sigmoid(out["graph_logit"]).detach().cpu())
        finals.append(torch.sigmoid(out["final_logit"]).detach().cpu())
        ranks.append(torch.sigmoid(out["rank_logit"]).detach().cpu())
        mixes.append(out["mix_alpha"].detach().cpu())
    feats = torch.stack(feats); graphs = torch.stack(graphs)
    finals = torch.stack(finals); ranks = torch.stack(ranks); mixes = torch.stack(mixes)
    # Use sample SD (correction=1), matching the manuscript definition with denominator T-1.
    final_mean = finals.mean(0).numpy(); final_std = finals.std(0, unbiased=True).numpy()
    rank_mean = ranks.mean(0).numpy(); rank_std = ranks.std(0, unbiased=True).numpy()
    return {
        "feature_score": feats.mean(0).numpy(),
        "graph_score": graphs.mean(0).numpy(),
        "pu_score": final_mean,
        "uncertainty": final_std,
        "confidence": 1.0 / (1.0 + final_std),
        "rank_head_mean": rank_mean,
        "rank_head_std": rank_std,
        "rank_score": rank_mean - uncertainty_penalty * rank_std,
        "mix_alpha": mixes.mean(0).numpy(),
    }



# =========================================================
# Training
# =========================================================
def train_one_stage(model, data, pi, stage_id=0,
                    epochs=200, lr=1e-3, weight_decay=5e-4,
                    patience=8, log_steps=10, Ks=(50, 100, 200),
                    beta=0.0, rank_weight=1.0,
                    save_dir=".", run_name="run", log_fn=None):
    def _log(s):
        print(s)
        if log_fn: log_fn(s)
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device); data = data.to(device)

    criterion = PULossWithRanking(pi=pi, beta=beta,
                                  rank_weight=rank_weight, margin=0.3,
                                  num_rank_pairs=2048, hard_negative_ratio=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20, min_lr=1e-5)

    best_state, best_val_key, bad = None, -1e9, 0
    history = []
    _log(f"[+] Start Stage-{stage_id} | pi={pi:.5f}")

    for e in range(1, epochs + 1):
        model.train(); optimizer.zero_grad()
        out = model(data, return_details=True)
        loss_pu, stats = criterion(out["final_logit"], out["rank_logit"],
                                   data.train_pos_mask, data.train_unlabeled_mask)
        loss = loss_pu

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if e == 1 or e % log_steps == 0:
            model.eval()
            preds = mc_dropout_predict(model, data, mc_times=10)
            val_m = compute_ranking_metrics(
                preds["rank_score"],
                eval_pos_mask=data.val_pos_mask,
                unlabeled_mask=data.unlabeled_mask,
                Ks=Ks
            )
            val_key = 0.6 * val_m.get("nDCG@100", 0.0) + 0.4 * val_m.get("Recall@100", 0.0)
            scheduler.step(val_key)
            rec = {"epoch": e, "stage": stage_id, "loss": float(loss.item()),
                   **{k: float(v.item()) for k, v in stats.items()},
                   **val_m, "val_key": float(val_key),
                   "lr": float(optimizer.param_groups[0]["lr"])}
            history.append(rec)
            _log(f"[S{stage_id}|{e:04d}] Loss:{rec['loss']:.4f} "
                 f"Rank:{rec.get('rank_loss',0):.3f} "
                 f"Rp+:{rec['Rp_pos']:.3f} NegRaw:{rec['neg_risk_raw']:.4f} "
                 f"R@100:{rec.get('Recall@100',0):.3f} nDCG@100:{rec.get('nDCG@100',0):.3f} "
                 f"MRR:{rec.get('MRR',0):.3f} LR:{rec['lr']:.5f}")
            if val_key > best_val_key:
                best_val_key = val_key
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, os.path.join(save_dir, f"{run_name}_best_stage_{stage_id}.pth"))
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    _log(f"[!] Stage-{stage_id} early stopping, best val_key={best_val_key:.4f}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val_key


def train_iterative(model, data, pi,
                    self_training_rounds=4, stage_epochs=200,
                    lr=1e-3, weight_decay=5e-4, patience=8, log_steps=10,
                    Ks=(50, 100, 200), beta=0.0,
                    rank_weight=1.0,
                    save_dir=".", run_name="run",
                    experiment_params=None, split_seed=None, train_seed=None):
    os.makedirs(save_dir, exist_ok=True)

    # Save parameters
    run_params = {
        "meta": {"run_name": run_name, "split_seed": split_seed,
                 "train_seed": train_seed,
                 "device": device,
                 "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        "training_params": {"pi": float(pi),
                            "self_training_rounds": int(self_training_rounds),
                            "stage_epochs": int(stage_epochs), "lr": float(lr),
                            "weight_decay": float(weight_decay), "patience": int(patience),
                            "beta": float(beta), 
                            "rank_weight": float(rank_weight)},
        "data_info": {"num_nodes": int(data.num_nodes),
                      "num_edges_main": int(data.edge_index.size(1)),
                      "train_pos": int(data.train_pos_mask.sum().item()),
                      "val_pos": int(data.val_pos_mask.sum().item()),
                      "test_pos": int(data.test_pos_mask.sum().item()),
                      "true_unlabeled": int(data.unlabeled_mask.sum().item()),
                      "training_unlabeled": int(data.train_unlabeled_mask.sum().item()),
                      "hidden_seeds_in_training_unlabeled": int((data.train_unlabeled_mask & data.hidden_seed_mask).sum().item())},
        "model_info": {"model_class": model.__class__.__name__,
                       "use_hetero": getattr(model, "use_hetero", False)},
    }
    if experiment_params is not None:
        run_params["experiment_params"] = experiment_params
    with open(os.path.join(save_dir, f"{run_name}_params.json"), "w", encoding="utf-8") as f:
        json.dump(run_params, f, ensure_ascii=False, indent=2)

    total_log_fp = open(os.path.join(save_dir, f"{run_name}_total_train_log.txt"),
                        "w", encoding="utf-8")
    def log(s):
        print(s); total_log_fp.write(s + "\n"); total_log_fp.flush()

    model = model.to(device); data = data.to(device)

    all_history = []
    g_best_state, g_best_val, g_best_stage = None, -1e9, -1

    log("=" * 80)
    log("[+] Start training (multi-stage warm-restart)")
    log(json.dumps(run_params, ensure_ascii=False, indent=2))
    log("=" * 80)
    if self_training_rounds > 0:
        log(f"[i] Will run {self_training_rounds} warm-restart stage(s) after base stage.")

    # Base stage
    model, hist, bv = train_one_stage(
        model, data, pi=pi, stage_id=0,
        epochs=stage_epochs, lr=lr, weight_decay=weight_decay, patience=patience,
        log_steps=log_steps, Ks=Ks, beta=beta, rank_weight=rank_weight,
        save_dir=save_dir, run_name=run_name, log_fn=log)
    all_history.extend(hist)
    if bv > g_best_val:
        g_best_val = bv; g_best_stage = 0
        g_best_state = copy.deepcopy(model.state_dict())
        torch.save(g_best_state, os.path.join(save_dir, f"{run_name}_best_global.pth"))

    # Warm restart stages
    for stage in range(1, self_training_rounds + 1):
        log(f"\n[+] Warm-restart Stage-{stage}")
        model, hist, bv = train_one_stage(
            model, data, pi=pi, stage_id=stage,
            epochs=stage_epochs, lr=lr, weight_decay=weight_decay, patience=patience,
            log_steps=log_steps, Ks=Ks, beta=beta, 
            rank_weight=rank_weight, save_dir=save_dir, run_name=run_name, log_fn=log)
        all_history.extend(hist)
        if bv > g_best_val:
            g_best_val = bv; g_best_stage = stage
            g_best_state = copy.deepcopy(model.state_dict())
            torch.save(g_best_state, os.path.join(save_dir, f"{run_name}_best_global.pth"))

    if g_best_state is not None:
        model.load_state_dict(g_best_state)
        log(f"[+] Reloaded global best from Stage-{g_best_stage}, val_key={g_best_val:.4f}")

    df_h = pd.DataFrame(all_history)
    df_h.to_csv(os.path.join(save_dir, f"{run_name}_training_history.csv"), index=False)

    model.eval()
    final_preds = mc_dropout_predict(model, data, mc_times=30)
    test_m = compute_ranking_metrics(
        final_preds["rank_score"],
        eval_pos_mask=data.test_pos_mask,
        unlabeled_mask=data.unlabeled_mask,
        Ks=Ks
    )
    log(f"[+] Final test metrics: {test_m}")
    total_log_fp.close()
    return model, all_history, test_m, final_preds


def compute_tier_recovery(scores: np.ndarray, test_pos_mask: torch.Tensor,
                          unlabeled_mask: torch.Tensor,
                          tier_groups: np.ndarray,
                          Ks=(50, 100, 200)) -> List[Dict]:
    """
    Descriptive recovery by evidence tier within the same overall test candidate universe.

    This intentionally avoids tier-specific AUROC/AUPRC, which can be unstable for small
    tiers and would incorrectly treat held-out positives from other tiers as negatives.
    """
    test_pos = test_pos_mask.detach().cpu().numpy().astype(bool)
    unlabeled = unlabeled_mask.detach().cpu().numpy().astype(bool)
    candidate = test_pos | unlabeled
    cand = np.where(candidate)[0]
    order = cand[np.argsort(-scores[cand])]
    rank_of = np.full(len(scores), np.nan, dtype=float)
    rank_of[order] = np.arange(1, len(order) + 1, dtype=float)

    rows = []
    tiers = sorted({str(t) for t in tier_groups[test_pos] if str(t) not in {"", "Unspecified"}})
    for tier in tiers:
        idx = np.where(test_pos & (tier_groups == tier))[0]
        if len(idx) == 0:
            continue
        ranks = rank_of[idx]
        row = {
            "tier_group": tier,
            "n_test_pos": int(len(idx)),
            "median_rank": float(np.nanmedian(ranks)),
            "mean_rank": float(np.nanmean(ranks)),
            "median_rank_percentile": float(np.nanmedian(ranks / max(len(order), 1))),
        }
        for K in Ks:
            hits = int(np.sum(ranks <= K))
            row[f"Hits@{K}"] = hits
            row[f"Recall@{K}"] = float(hits / len(idx))
        rows.append(row)
    return rows


def bootstrap_mean_ci(values, ci=0.95, n_boot=10000, seed=20260930):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    # Chunking keeps memory small even if n_boot is increased later.
    chunk = 1000
    filled = 0
    while filled < n_boot:
        b = min(chunk, n_boot - filled)
        sample_idx = rng.integers(0, n, size=(b, n))
        means[filled:filled + b] = values[sample_idx].mean(axis=1)
        filled += b
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def summarize_metrics_df(metrics_df: pd.DataFrame,
                         ci=0.95, n_boot=10000,
                         seed=20260930) -> pd.DataFrame:
    """Summarize repeated resampling with both bootstrap and t-based 95% CIs."""
    excluded = {"repeat", "split_seed", "train_seed"}
    metric_cols = [c for c in metrics_df.columns if c not in excluded]
    rows = []
    for j, metric in enumerate(metric_cols):
        vals = pd.to_numeric(metrics_df[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        boot_lo, boot_hi = bootstrap_mean_ci(vals, ci=ci, n_boot=n_boot, seed=seed + j)
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        sem = float(sd / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        if len(vals) > 1 and SCIPY_AVAILABLE:
            alpha = 1.0 - ci
            crit = float(student_t.ppf(1.0 - alpha / 2.0, df=len(vals) - 1))
            t_lo = mean - crit * sem
            t_hi = mean + crit * sem
        else:
            t_lo = mean
            t_hi = mean
        rows.append({
            "metric": metric,
            "n_repeats": int(len(vals)),
            "mean": mean,
            "std": sd,
            "median": float(np.median(vals)),
            "sem": sem,
            # Keep ci95_low/high as bootstrap CI for backward compatibility.
            "ci95_low": boot_lo,
            "ci95_high": boot_hi,
            "ci95_boot_low": boot_lo,
            "ci95_boot_high": boot_hi,
            "ci95_t_low": float(t_lo),
            "ci95_t_high": float(t_hi),
            "split_q025": float(np.quantile(vals, 0.025)),
            "split_q975": float(np.quantile(vals, 0.975)),
        })
    return pd.DataFrame(rows)


def _holm_adjust(p_values: List[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return out.tolist()
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p[idx])
        running = max(running, adj)
        out[idx] = running
    return out.tolist()


def run_paired_comparisons(main_df: pd.DataFrame,
                           baseline_specs: List[str],
                           output_csv: str,
                           bootstrap_reps: int = 10000) -> Optional[pd.DataFrame]:
    """
    Optional paired statistics against baseline per-split CSV files.

    Each spec is NAME=/path/to/per_split_metrics.csv and must contain split_seed plus
    metric columns. Wilcoxon signed-rank tests are paired on split_seed; Holm correction
    is applied across all generated comparisons.
    """
    if not baseline_specs:
        return None
    if not SCIPY_AVAILABLE:
        print("[!] scipy is unavailable; paired Wilcoxon tests were skipped.")
        return None

    id_col = "split_seed"
    if id_col not in main_df.columns:
        raise ValueError("Main per-split results lack split_seed")
    rows = []
    for spec in baseline_specs:
        if "=" not in spec:
            raise ValueError(f"--baseline_metrics must be NAME=path, got: {spec}")
        name, path = spec.split("=", 1)
        base = pd.read_csv(path)
        if id_col not in base.columns:
            raise ValueError(f"Baseline {name} lacks split_seed: {path}")
        merged = main_df.merge(base, on=id_col, suffixes=("_main", "_base"))
        main_metrics = [c for c in main_df.columns if c not in {"repeat", "split_seed", "train_seed"}]
        common = [m for m in main_metrics if m in base.columns]
        for j, metric in enumerate(common):
            a = pd.to_numeric(merged[f"{metric}_main"], errors="coerce").to_numpy(float)
            b = pd.to_numeric(merged[f"{metric}_base"], errors="coerce").to_numpy(float)
            good = np.isfinite(a) & np.isfinite(b)
            a, b = a[good], b[good]
            if len(a) < 2:
                continue
            diff = a - b
            try:
                if np.allclose(diff, 0):
                    pval = 1.0
                else:
                    pval = float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
            except Exception:
                pval = np.nan
            lo, hi = bootstrap_mean_ci(diff, n_boot=bootstrap_reps,
                                       seed=20261001 + j + len(rows))
            rows.append({
                "baseline": name.strip(),
                "metric": metric,
                "n_pairs": int(len(diff)),
                "mean_paired_difference_main_minus_baseline": float(np.mean(diff)),
                "median_paired_difference_main_minus_baseline": float(np.median(diff)),
                "difference_ci95_low": lo,
                "difference_ci95_high": hi,
                "wilcoxon_p": pval,
            })

    out = pd.DataFrame(rows)
    if len(out):
        out["holm_adjusted_p"] = _holm_adjust(out["wilcoxon_p"].tolist())
        out.to_csv(output_csv, index=False)
        print(f"[+] Saved paired statistical comparisons: {output_csv}")
    return out


# =========================================================
# Main
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="PoplarCold-PUGAT training with grouped splits, hidden-seed robustness, split manifests, and CI/statistics."
    )
    repo_dir = Path(__file__).resolve().parent
    default_data_dir = repo_dir / "data" / "preprocessed_label_independent_ppi800_corr0.75"
    p.add_argument("--data_dir", type=str, default=str(default_data_dir))
    p.add_argument("--save_root", type=str, default="")

    # Audited seed metadata / reproducible split definition
    p.add_argument("--seed_metadata", type=str, default="",
                   help="Audited seed xlsx/csv containing Pt_Gene_ID, Grouping_Key, Evidence_Tier")
    p.add_argument("--metadata_sheet", type=str, default="Supplementary_Data_1_Revised")
    p.add_argument("--gene_col", type=str, default="Pt_Gene_ID")
    p.add_argument("--group_col", type=str, default="Grouping_Key")
    p.add_argument("--tier_col", type=str, default="Evidence_Tier")
    p.add_argument("--split_mode", type=str, default="grouped", choices=["grouped", "random"],
                   help="grouped keeps each Grouping_Key entirely in one partition")
    p.add_argument("--group_split_strategy", type=str, default="random_search",
                   choices=["legacy_greedy", "random_search"],
                   help=("Grouped split allocator. random_search fixes near-target group counts and "
                         "selects the best split using gene-count/evidence-tier balance only; "
                         "legacy_greedy reproduces the earlier gene-count-only allocator."))
    p.add_argument("--group_split_trials", type=int, default=20000,
                   help="Number of pre-model candidate grouped assignments evaluated per repeat")
    p.add_argument("--group_gene_balance_weight", type=float, default=1.0,
                   help="Weight on positive-gene count balance in random_search grouped splitting")
    p.add_argument("--group_tier_balance_weight", type=float, default=1.0,
                   help="Weight on collapsed evidence-tier balance in random_search grouped splitting")
    p.add_argument("--split_manifest_in", type=str, default="",
                   help="Reuse an existing split_manifest.csv; recommended for all baseline/robustness runs")
    p.add_argument("--hidden_seed_as_unlabeled", action="store_true",
                   help="Treat validation/test seed genes as unlabeled in nnPU and ranking loss during training")
    p.add_argument("--manifest_only", action="store_true",
                   help="Generate/validate split_manifest.csv and exit before model training")

    # Graph
    p.add_argument("--coexp_ratio", type=float, default=0.5)
    p.add_argument("--ppi_ratio", type=float, default=0.5)
    p.add_argument("--no_coexp", action="store_true")
    p.add_argument("--no_ppi", action="store_true")
    p.add_argument("--graph_mode", type=str, default="hetero", choices=["hetero", "merged"],
                   help="hetero = separate coexpression/PPI GATv2 branches (intended main model)")

    # PU prior
    p.add_argument("--pi_method", type=str, default="manual",
                   choices=["fixed", "manual", "tice"],
                   help="Manual mode keeps the working prior fixed across robustness conditions")
    p.add_argument("--pi_override", type=float, default=0.013,
                   help="Positive-class working prior for manual mode")

    # Model
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--proj_dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)

    # Training / repeated resampling
    p.add_argument("--seed", type=int, default=42,
                   help="Base split seed; repeat r uses split_seed = seed + r")
    p.add_argument("--train_seed_base", type=int, default=10042,
                   help="Base training seed used when --train_seed_mode independent")
    p.add_argument("--train_seed_mode", type=str, default="independent",
                   choices=["independent", "same_as_split", "legacy_stream"],
                   help=("independent: reset training RNG per repeat using train_seed_base + repeat-1; "
                         "same_as_split: use split_seed for training RNG; "
                         "legacy_stream: seed once at program start and do not reset between repeats"))
    p.add_argument("--num_repeats", type=int, default=10)
    p.add_argument("--train_ratio", type=float, default=0.5)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--self_training_rounds", type=int, default=4)
    p.add_argument("--stage_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--rank_weight", type=float, default=1.0)

    # Statistical output
    p.add_argument("--ci_bootstrap_reps", type=int, default=10000)
    p.add_argument("--baseline_metrics", action="append", default=[],
                   help="Optional paired comparison, repeatable: NAME=/path/to/per_split_metrics.csv")
    return p.parse_args()


def collect_env_info():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        info["git_commit"] = subprocess.getoutput("git rev-parse HEAD").strip()
    except Exception:
        info["git_commit"] = "unknown"
    try:
        import torch_geometric
        info["torch_geometric"] = torch_geometric.__version__
    except Exception:
        info["torch_geometric"] = "unknown"
    return info


def main():
    args = parse_args()
    set_seed(args.seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.save_root:
        save_root = args.save_root
    else:
        save_root = str(Path(__file__).resolve().parent / "results" / f"main_{ts}")
    os.makedirs(save_root, exist_ok=True)

    # Global log
    global_log_fp = open(os.path.join(save_root, "global_run.log"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, global_log_fp)
    sys.stderr = Tee(sys.__stderr__, global_log_fp)
    with open(os.path.join(save_root, "env_info.json"), "w", encoding="utf-8") as f:
        json.dump(collect_env_info(), f, ensure_ascii=False, indent=2)

    print(f"[+] save_root: {save_root}")
    print(f"[+] args: {vars(args)}")

    bundle = load_preprocessed(args.data_dir)

    # ---------------------------------------------------------
    # Audited seed metadata: grouping/evidence only; y.npy is authoritative
    # ---------------------------------------------------------
    metadata = None
    preprocessing_summary = {}
    preprocessing_summary_path = Path(args.data_dir) / "summary.json"
    if preprocessing_summary_path.exists():
        with open(preprocessing_summary_path, "r", encoding="utf-8") as f:
            preprocessing_summary = json.load(f)
        with open(os.path.join(save_root, "preprocessing_summary_used.json"), "w", encoding="utf-8") as f:
            json.dump(preprocessing_summary, f, ensure_ascii=False, indent=2)
        print(f"[i] Loaded preprocessing summary: {preprocessing_summary_path}")

    if args.seed_metadata:
        metadata = load_seed_metadata(
            args.seed_metadata, sheet_name=args.metadata_sheet,
            gene_col=args.gene_col, group_col=args.group_col, tier_col=args.tier_col
        )
        bundle = attach_seed_metadata(
            bundle, metadata,
            gene_col=args.gene_col, group_col=args.group_col, tier_col=args.tier_col
        )

        # Per-gene metadata/graph/label coverage audit.
        raw_col = f"{args.gene_col}_raw"
        coverage_cols = [c for c in [raw_col, args.gene_col, args.group_col, args.tier_col] if c in metadata.columns]
        coverage = metadata[coverage_cols].copy()
        graph_genes = set(bundle["nodes"])
        y_arr = np.asarray(bundle["y"], dtype=np.int64)
        current_pos = set(np.asarray(bundle["nodes"], dtype=object)[y_arr == 1].tolist())
        coverage["matched_to_graph"] = coverage[args.gene_col].isin(graph_genes)
        coverage["preprocessed_y_positive"] = coverage[args.gene_col].isin(current_pos)
        coverage["group_key_present"] = coverage[args.group_col].fillna("").astype(str).str.strip().ne("")
        coverage["tier_group"] = coverage[args.tier_col].map(collapse_evidence_tier)
        coverage.to_csv(os.path.join(save_root, "seed_metadata_coverage.csv"), index=False)

        # Single-row summary, including the requested matched_to_graph count.
        match_summary = dict(bundle.get("seed_metadata_match_counts", {}))
        for key in [
            "n_genes_before_filter", "n_genes_after_filter",
            "n_known_positives_before_filter", "n_known_positives_after_filter",
            "n_edgeless_nodes_removed", "n_edgeless_known_positives_removed",
            "n_edgeless_unlabeled_removed",
        ]:
            if key in preprocessing_summary:
                match_summary[f"preprocess_{key}"] = preprocessing_summary[key]
        pd.DataFrame([match_summary]).to_csv(
            os.path.join(save_root, "seed_metadata_match_summary.csv"), index=False
        )

        # Grouped splitting requires metadata for every positive remaining after preprocessing.
        # Save the audit CSVs first, then fail loudly if the metadata cannot support a valid grouped split.
        if args.split_mode == "grouped":
            missing_meta = bundle.get("seed_positive_missing_metadata", [])
            missing_group = bundle.get("seed_positive_missing_group", [])
            if missing_meta:
                raise ValueError(
                    f"{len(missing_meta)} y=1 genes from preprocessing are absent from seed metadata. "
                    f"See seed_metadata_match_summary.csv. Examples: {missing_meta[:10]}"
                )
            if missing_group:
                raise ValueError(
                    f"{len(missing_group)} preprocessed positive genes lack {args.group_col}. "
                    f"Examples: {missing_group[:10]}"
                )
    else:
        if args.split_mode == "grouped":
            raise ValueError(f"Grouped splitting requires --seed_metadata with {args.group_col} (or At_Gene_ID fallback).")
        # Empty metadata arrays keep downstream reporting code simple.
        bundle["node_group_key"] = np.array(["" for _ in bundle["nodes"]], dtype=object)
        bundle["node_evidence_tier"] = np.array(["" for _ in bundle["nodes"]], dtype=object)
        bundle["node_tier_group"] = np.array(["" for _ in bundle["nodes"]], dtype=object)

    n_pos = int((np.asarray(bundle["y"]) == 1).sum())
    n_nodes = len(bundle["y"])
    print(f"[i] Positive set from preprocessing y.npy: {n_pos}/{n_nodes} nodes")
    if preprocessing_summary:
        reported_after = preprocessing_summary.get("n_known_positives_after_filter")
        if reported_after is not None and int(reported_after) != n_pos:
            raise ValueError(
                "Preprocessing summary and y.npy disagree: "
                f"summary n_known_positives_after_filter={reported_after}, y.npy positives={n_pos}"
            )

    # ---------------------------------------------------------
    # Split manifest: generate once, then reuse for all models
    # ---------------------------------------------------------
    pos_genes = [bundle["nodes"][i] for i in np.where(np.asarray(bundle["y"]) == 1)[0]]
    if args.split_manifest_in:
        manifest = pd.read_csv(args.split_manifest_in)
        # Normalize types because CSV readers may infer repeat/seed as floats.
        manifest["repeat"] = pd.to_numeric(manifest["repeat"], errors="raise").astype(int)
        manifest["split_seed"] = pd.to_numeric(manifest["split_seed"], errors="raise").astype(int)
        validate_split_manifest(manifest, pos_genes, split_mode=args.split_mode)
        print(f"[i] Reusing split manifest: {args.split_manifest_in}")
    else:
        manifest = generate_split_manifest(
            bundle=bundle, metadata=metadata,
            split_mode=args.split_mode, num_repeats=args.num_repeats,
            seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
            gene_col=args.gene_col, group_col=args.group_col, tier_col=args.tier_col,
            group_split_strategy=args.group_split_strategy,
            group_split_trials=args.group_split_trials,
            group_gene_balance_weight=args.group_gene_balance_weight,
            group_tier_balance_weight=args.group_tier_balance_weight,
        )
        print(f"[i] Generated {args.split_mode} split manifest with {args.num_repeats} repeats")

    manifest_path = os.path.join(save_root, "split_manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    save_split_balance_reports(
        manifest=manifest,
        save_root=save_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    split_config = {
        "data_dir": args.data_dir,
        "seed_metadata": args.seed_metadata,
        "metadata_sheet": args.metadata_sheet,
        "gene_col": args.gene_col,
        "group_col": args.group_col,
        "tier_col": args.tier_col,
        "split_mode": args.split_mode,
        "group_split_strategy": args.group_split_strategy,
        "group_split_trials": args.group_split_trials,
        "group_gene_balance_weight": args.group_gene_balance_weight,
        "group_tier_balance_weight": args.group_tier_balance_weight,
        "num_repeats": args.num_repeats,
        "base_seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "n_positive_nodes_used": n_pos,
        "n_graph_nodes": n_nodes,
        "split_manifest_saved": manifest_path,
    }
    with open(os.path.join(save_root, "split_config.json"), "w", encoding="utf-8") as f:
        json.dump(split_config, f, ensure_ascii=False, indent=2)

    if args.manifest_only:
        print("[+] --manifest_only requested; split manifest validated and saved. Exiting before training.")
        global_log_fp.close()
        return

    # ---------------------------------------------------------
    # Class prior: keep fixed across robustness conditions if manual
    # ---------------------------------------------------------
    pi = estimate_pi(args.pi_method, n_pos=n_pos, n_nodes=n_nodes,
                     pi_override=args.pi_override,
                     X=bundle["X"], pos_mask=(np.asarray(bundle["y"]) == 1))
    print(f"[i] pi by '{args.pi_method}': {pi:.5f} "
          f"(seed frequency would be {n_pos / n_nodes:.5f})")

    exp_params = vars(args).copy()
    exp_params["pi_used"] = pi
    exp_params["timestamp"] = ts
    exp_params["n_positive_nodes_used"] = n_pos
    exp_params["n_manifest_repeats"] = int(manifest["repeat"].nunique())
    exp_params["split_manifest_saved"] = manifest_path
    with open(os.path.join(save_root, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(exp_params, f, ensure_ascii=False, indent=2)

    use_hetero = (args.graph_mode == "hetero")

    def model_builder(data: Data):
        return SingleOmicPUGAT(
            in_dim=data.x.size(1),
            hidden_dim=args.hidden_dim, proj_dim=args.proj_dim,
            heads=args.heads, dropout=args.dropout,
            use_hetero=use_hetero,
        )

    all_metrics: List[Dict] = []
    all_tier_rows: List[Dict] = []
    score_sum = np.zeros(n_nodes, dtype=np.float64)
    score_sq_sum = np.zeros(n_nodes, dtype=np.float64)
    uncertainty_sum = np.zeros(n_nodes, dtype=np.float64)

    repeats = sorted(manifest["repeat"].unique().tolist())
    for rep in repeats:
        manifest_repeat = manifest.loc[manifest["repeat"] == rep].copy()
        seeds_this = manifest_repeat["split_seed"].unique()
        if len(seeds_this) != 1:
            raise ValueError(f"Repeat {rep} has multiple split_seed values: {seeds_this}")
        split_seed = int(seeds_this[0])

        if args.train_seed_mode == "independent":
            train_seed = int(args.train_seed_base + (int(rep) - 1))
            set_seed(train_seed)
        elif args.train_seed_mode == "same_as_split":
            train_seed = split_seed
            set_seed(train_seed)
        elif args.train_seed_mode == "legacy_stream":
            train_seed = None
            # Do not reset global RNG here. main() seeded it once at program start.
        else:
            raise ValueError(f"Unknown train_seed_mode: {args.train_seed_mode}")

        print("\n" + "-" * 100)
        print(f"[+] Repeat {rep}/{len(repeats)} | split_seed={split_seed} | "
              f"train_seed={train_seed if train_seed is not None else 'legacy_stream'} | "
              f"split_mode={args.split_mode} | hidden_seed_as_unlabeled={args.hidden_seed_as_unlabeled}")
        print("-" * 100)

        data_r = build_data_for_split(
            bundle, manifest_repeat=manifest_repeat,
            hidden_seed_as_unlabeled=args.hidden_seed_as_unlabeled,
            use_coexp=not args.no_coexp, use_ppi=not args.no_ppi,
            coexp_ratio=args.coexp_ratio, ppi_ratio=args.ppi_ratio,
            use_hetero=use_hetero,
        )

        train_seed_tag = str(train_seed) if train_seed is not None else "legacy_stream"
        run_name = f"repeat_{int(rep):02d}_split_{split_seed}_train_{train_seed_tag}"
        sd = os.path.join(save_root, run_name)
        os.makedirs(sd, exist_ok=True)
        model = model_builder(data_r)
        _, _, test_m, final_preds = train_iterative(
            model=model, data=data_r, pi=pi,
            self_training_rounds=args.self_training_rounds,
            stage_epochs=args.stage_epochs, lr=args.lr,
            weight_decay=args.weight_decay, patience=args.patience,
            log_steps=args.log_steps, Ks=(50, 100, 200),
            beta=args.beta,
            rank_weight=args.rank_weight,
            save_dir=sd, run_name=run_name,
            experiment_params=exp_params, split_seed=split_seed,
            train_seed=train_seed
        )

        rank_score = np.asarray(final_preds["rank_score"], dtype=float)
        uncertainty = np.asarray(final_preds["uncertainty"], dtype=float)
        score_sum += rank_score
        score_sq_sum += rank_score ** 2
        uncertainty_sum += uncertainty

        # Save per-repeat novel-candidate ranking and audit columns.
        rank_df = pd.DataFrame({
            "gene_id": bundle["nodes"],
            "rank_score": rank_score,
            "pu_score": final_preds["pu_score"],
            "uncertainty": uncertainty,
            "group_key": bundle.get("node_group_key", np.array([""] * n_nodes, dtype=object)),
            "evidence_tier": bundle.get("node_evidence_tier", np.array([""] * n_nodes, dtype=object)),
            "tier_group": bundle.get("node_tier_group", np.array([""] * n_nodes, dtype=object)),
            "is_train_pos": data_r.train_pos_mask.detach().cpu().numpy(),
            "is_val_pos": data_r.val_pos_mask.detach().cpu().numpy(),
            "is_test_pos": data_r.test_pos_mask.detach().cpu().numpy(),
            "is_training_unlabeled": data_r.train_unlabeled_mask.detach().cpu().numpy(),
        })
        known_pos_mask = (np.asarray(bundle["y"]) == 1)
        novel_rank_df = (
            rank_df.loc[~known_pos_mask]
            .sort_values("rank_score", ascending=False)
            .reset_index(drop=True)
        )
        novel_rank_df.insert(0, "rank", np.arange(1, len(novel_rank_df) + 1))
        novel_rank_df.to_csv(os.path.join(sd, f"{run_name}_ranking.csv"), index=False)

        metric_row = {
            "repeat": int(rep),
            "split_seed": split_seed,
            "train_seed": train_seed,
            **test_m
        }
        all_metrics.append(metric_row)
        print(f"[i] split_seed={split_seed} train_seed="
              f"{train_seed if train_seed is not None else 'legacy_stream'} TEST: {test_m}")

        # Descriptive evidence-tier recovery (same candidate universe as the main test).
        tier_rows = compute_tier_recovery(
            rank_score,
            test_pos_mask=data_r.test_pos_mask,
            unlabeled_mask=data_r.unlabeled_mask,
            tier_groups=np.asarray(bundle.get("node_tier_group", [""] * n_nodes), dtype=object),
            Ks=(50, 100, 200)
        )
        for row in tier_rows:
            row.update({
                "repeat": int(rep),
                "split_seed": split_seed,
                "train_seed": train_seed
            })
            all_tier_rows.append(row)

    # ---------------------------------------------------------
    # Repeated-split metrics + bootstrap 95% CI
    # ---------------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics)
    per_split_path = os.path.join(save_root, "per_split_metrics.csv")
    metrics_df.to_csv(per_split_path, index=False)

    summary_df = summarize_metrics_df(
        metrics_df, ci=0.95, n_boot=args.ci_bootstrap_reps, seed=args.seed + 100000
    )
    summary_path = os.path.join(save_root, "summary_metrics_95CI.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print(f"[+] Repeated {args.split_mode}-resampling summary")
    for _, row in summary_df.iterrows():
        print(f"{row['metric']}: {row['mean']:.4f} ± {row['std']:.4f} "
              f"(95% bootstrap CI {row['ci95_boot_low']:.4f}–{row['ci95_boot_high']:.4f}; "
              f"95% t-CI {row['ci95_t_low']:.4f}–{row['ci95_t_high']:.4f})")
    print("=" * 100)

    txt_path = os.path.join(save_root, "final_repeated_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Repeated {args.split_mode}-resampling results\n")
        f.write("NOTE: intervals summarize repeated resampling variability; they are not independent biological replicates.\n\n")
        for _, row in summary_df.iterrows():
            f.write(
                f"{row['metric']}: mean={row['mean']:.6f}, sd={row['std']:.6f}, "
                f"95% bootstrap CI=[{row['ci95_boot_low']:.6f}, {row['ci95_boot_high']:.6f}], "
                f"95% t-CI=[{row['ci95_t_low']:.6f}, {row['ci95_t_high']:.6f}], "
                f"n={int(row['n_repeats'])}\n"
            )
        f.write("\nPer-split metrics are in per_split_metrics.csv\n")

    if all_tier_rows:
        tier_df = pd.DataFrame(all_tier_rows)
        tier_path = os.path.join(save_root, "per_split_evidence_tier_recovery.csv")
        tier_df.to_csv(tier_path, index=False)
        # Pooled descriptive hit rates are less fragile than per-tier AUROC/AUPRC for tiny tiers.
        pooled_rows = []
        for tier, sub in tier_df.groupby("tier_group"):
            row = {
                "tier_group": tier,
                "test_occurrences_across_repeats": int(sub["n_test_pos"].sum()),
                "number_of_repeats_with_test_genes": int(len(sub)),
                "median_of_split_median_rank": float(sub["median_rank"].median()),
            }
            for K in (50, 100, 200):
                denom = float(sub["n_test_pos"].sum())
                row[f"pooled_Recall@{K}"] = float(sub[f"Hits@{K}"].sum() / denom) if denom > 0 else np.nan
            pooled_rows.append(row)
        pd.DataFrame(pooled_rows).to_csv(
            os.path.join(save_root, "evidence_tier_pooled_recovery.csv"), index=False
        )

    # ---------------------------------------------------------
    # Final ensemble ranking across exactly these repeated runs
    # ---------------------------------------------------------
    n_runs = len(repeats)
    score_mean = score_sum / max(n_runs, 1)
    score_var = np.maximum(score_sq_sum / max(n_runs, 1) - score_mean ** 2, 0.0)
    score_sd = np.sqrt(score_var)
    ensemble = pd.DataFrame({
        "gene_id": bundle["nodes"],
        "mean_rank_score": score_mean,
        "sd_rank_score_across_repeats": score_sd,
        "mean_mc_uncertainty": uncertainty_sum / max(n_runs, 1),
        "group_key": bundle.get("node_group_key", np.array([""] * n_nodes, dtype=object)),
        "evidence_tier": bundle.get("node_evidence_tier", np.array([""] * n_nodes, dtype=object)),
        "tier_group": bundle.get("node_tier_group", np.array([""] * n_nodes, dtype=object)),
        "is_known_seed": (np.asarray(bundle["y"]) == 1),
    })
    ensemble = (
        ensemble.loc[~ensemble["is_known_seed"]]
        .sort_values("mean_rank_score", ascending=False)
        .reset_index(drop=True)
    )
    ensemble.insert(0, "rank", np.arange(1, len(ensemble) + 1))
    ensemble_path = os.path.join(save_root, "ensemble_ranking.csv")
    ensemble.to_csv(ensemble_path, index=False)
    print(f"[+] Saved ensemble ranking: {ensemble_path}")

    # Optional paired tests once baseline scripts have produced metrics on the same manifest.
    run_paired_comparisons(
        metrics_df, args.baseline_metrics,
        output_csv=os.path.join(save_root, "paired_comparisons.csv"),
        bootstrap_reps=args.ci_bootstrap_reps
    )

    print(f"[+] Saved per-split metrics: {per_split_path}")
    print(f"[+] Saved metric summary with 95% CI: {summary_path}")
    print(f"[+] Saved split manifest: {manifest_path}")
    global_log_fp.close()


if __name__ == "__main__":
    main()

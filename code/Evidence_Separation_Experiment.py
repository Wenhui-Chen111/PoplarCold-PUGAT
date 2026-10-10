#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evidence_Separation_Experiment.py

Purpose
-------
Prepare condition-specific preprocessing outputs for evidence-separation experiments
without rebuilding expression features or graphs, and summarize ranking stability after
Train_ColdPU.py has been run.

Why this is the correct controlled design
-----------------------------------------
Prepare_data.py uses a label-independent node-inclusion rule:
    keep = has_coexp_edge OR has_ppi_edge
Therefore X, the node universe, co-expression graph, PPI graph, and feature scaling do
not depend on which audited genes are labeled positive. For evidence separation, we
should hold those objects fixed and change only the positive labels y and the split
manifest. This is equivalent to rerunning the preprocessing label-construction step but
avoids recomputing identical graphs three times.

Conditions (based on Supplementary_Data_1_revised.xlsx)
-------------------------------------------------------
full                  : all audited seeds
no_expression         : exclude seeds whose label-defining Evidence_Source/Evidence_Type
                        is explicitly expression based
no_go                 : exclude seeds whose label-defining Evidence_Source/Evidence_Type
                        is explicitly GO-annotation based
no_expression_no_go   : exclude either of the two sources above

For the current audited spreadsheet the expected raw seed counts are:
    full=188, no_expression=170, no_go=96, no_expression_no_go=78
Actual y.npy positive counts can be slightly smaller if the label-independent graph-node
filter removes some seed genes. Both pre-filter and post-filter counts are reported.

Subcommands
-----------
prepare   Build condition data directories, seed tables, and condition manifests.
summarize Compare trained rankings on a fixed novel-candidate universe (all full seeds
          excluded in every condition) and collect reviewer-facing summary tables.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import kendalltau, spearmanr
    HAVE_SCIPY = True
except Exception:
    kendalltau = None
    spearmanr = None
    HAVE_SCIPY = False


CONDITIONS = ("full", "no_expression", "no_go", "no_expression_no_go")
EXPECTED_RAW_COUNTS = {
    "full": 188,
    "no_expression": 170,
    "no_go": 96,
    "no_expression_no_go": 78,
}

# Files actually used by Train_ColdPU.py plus reproducibility metadata.
STATIC_FILES = [
    "X.npy",
    "X_raw.npy",
    "cond_input.npy",
    "other_input.npy",
    "edges_coexp.npy",
    "edges_coexp_weight.npy",
    "edges_ppi.npy",
    "edges_ppi_weight.npy",
    "scaler_mean.npy",
    "scaler_scale.npy",
    "gene_ids.txt",
    "feature_names.txt",
    "binary_idx.txt",
    "condition_meta.json",
    "feature_manifest.csv",
]


def norm_gene_id(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().split()[0]
    if not s or s.lower() == "nan":
        return None
    if s.startswith("Potri."):
        return re.sub(r"v\d+$", "", s)
    if s.startswith("POPTR_"):
        return "Potri." + re.sub(r"v\d+$", "", s[len("POPTR_"):])
    if s.startswith("POPTR."):
        return "Potri." + re.sub(r"v\d+$", "", s[len("POPTR."):])
    if s.startswith("Potri_"):
        return "Potri." + re.sub(r"v\d+$", "", s[len("Potri_"):])
    return s


def read_gene_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [norm_gene_id(x.strip()) or x.strip() for x in f if x.strip()]


def read_seed_metadata(path: Path, sheet: str) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet)
    elif path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported metadata format: {path.suffix}")
    if "Pt_Gene_ID" not in df.columns:
        raise ValueError("Seed metadata must contain Pt_Gene_ID")
    df = df.copy()
    df["Pt_Gene_ID"] = df["Pt_Gene_ID"].apply(norm_gene_id)
    df = df[df["Pt_Gene_ID"].notna()].copy()
    if df["Pt_Gene_ID"].duplicated().any():
        dup = df.loc[df["Pt_Gene_ID"].duplicated(keep=False), "Pt_Gene_ID"].tolist()
        raise ValueError(f"Duplicate Pt_Gene_ID values after normalization: {dup[:10]}")
    return df.reset_index(drop=True)


def text_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    return df[col].fillna("").astype(str).str.strip()


def derive_evidence_flags(meta: pd.DataFrame) -> pd.DataFrame:
    """Derive transparent label-defining evidence flags from audited columns."""
    out = meta.copy()
    source = text_col(out, "Evidence_Source")
    etype = text_col(out, "Evidence_Type")
    source_l = source.str.lower()
    etype_l = etype.str.lower()

    # Expression-based label evidence: explicit expression evidence only.
    # GO evidence code IEP is intentionally NOT folded into this category.
    expression_based = (
        source_l.str.contains("expression", regex=False)
        | etype_l.str.contains("expression evidence", regex=False)
    )

    # GO-based label evidence: the audited source/type itself is GO-annotation based.
    # A literature seed is not removed merely because a GO term is recorded elsewhere.
    go_based = (
        source_l.str.contains("go annotation", regex=False)
        | source_l.str.contains("go phylogenetic annotation", regex=False)
        | etype_l.str.contains("curated go", regex=False)
        | etype_l.str.contains("go annotation", regex=False)
        | etype_l.str.contains("cold-associated go", regex=False)
        | etype_l.str.contains("cold-acclimation go", regex=False)
        | etype_l.str.contains("freezing-response go", regex=False)
    )

    out["flag_expression_label"] = expression_based.astype(int)
    out["flag_go_label"] = go_based.astype(int)
    out["keep_full"] = 1
    out["keep_no_expression"] = (~expression_based).astype(int)
    out["keep_no_go"] = (~go_based).astype(int)
    out["keep_no_expression_no_go"] = (~(expression_based | go_based)).astype(int)

    out["exclusion_reason_no_expression"] = np.where(
        expression_based,
        "label-defining evidence explicitly uses expression evidence",
        "",
    )
    out["exclusion_reason_no_go"] = np.where(
        go_based,
        "label-defining evidence explicitly uses GO annotation evidence",
        "",
    )
    out["exclusion_reason_no_expression_no_go"] = ""
    out.loc[expression_based, "exclusion_reason_no_expression_no_go"] = "expression-based label evidence"
    out.loc[go_based & ~expression_based, "exclusion_reason_no_expression_no_go"] = "GO-based label evidence"
    out.loc[go_based & expression_based, "exclusion_reason_no_expression_no_go"] = "expression- and GO-based label evidence"
    return out


def condition_seed_table(flagged: pd.DataFrame, condition: str) -> pd.DataFrame:
    col = {
        "full": "keep_full",
        "no_expression": "keep_no_expression",
        "no_go": "keep_no_go",
        "no_expression_no_go": "keep_no_expression_no_go",
    }[condition]
    sub = flagged.loc[pd.to_numeric(flagged[col], errors="coerce").fillna(0).astype(int) == 1].copy()
    sub.insert(0, "evidence_condition", condition)
    return sub.reset_index(drop=True)


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    if not src.exists():
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(mode)


def load_base_node_audit(base_dir: Path) -> Optional[pd.DataFrame]:
    p = base_dir / "node_filter_audit.csv"
    if not p.exists():
        return None
    x = pd.read_csv(p)
    if "gene_id" in x.columns:
        x["gene_id"] = x["gene_id"].apply(norm_gene_id)
    return x


def prepare_condition_dir(
    base_dir: Path,
    target_dir: Path,
    seed_table: pd.DataFrame,
    condition: str,
    mode: str,
) -> Dict[str, int]:
    target_dir.mkdir(parents=True, exist_ok=True)
    nodes = read_gene_ids(base_dir / "gene_ids.txt")
    node_set = set(nodes)
    raw_seed_genes = [norm_gene_id(x) for x in seed_table["Pt_Gene_ID"].tolist()]
    raw_seed_genes = [x for x in raw_seed_genes if x]
    raw_seed_set = set(raw_seed_genes)
    y = np.asarray([1 if g in raw_seed_set else 0 for g in nodes], dtype=np.int64)
    graph_seed_set = set(np.asarray(nodes, dtype=object)[y == 1].tolist())

    # Static model inputs are deliberately frozen across all evidence conditions.
    for name in STATIC_FILES:
        src = base_dir / name
        if src.exists():
            copy_or_link(src, target_dir / name, mode)
    np.save(target_dir / "y.npy", y)

    # Condition-specific seed table used as --seed_metadata by the trainer.
    seed_table.to_csv(target_dir / "seed_metadata.csv", index=False)

    # Update labels in human-readable reports when the corresponding base report exists.
    feat_path = base_dir / "features_raw_table.csv"
    if feat_path.exists():
        feat = pd.read_csv(feat_path)
        if "gene_id" in feat.columns:
            feat["gene_id"] = feat["gene_id"].apply(norm_gene_id)
            feat["y"] = feat["gene_id"].isin(graph_seed_set).astype(int)
        feat.to_csv(target_dir / "features_raw_table.csv", index=False)

    base_audit = load_base_node_audit(base_dir)
    missing_from_graph = sorted(raw_seed_set - node_set)
    if base_audit is not None:
        a = base_audit.copy()
        a["is_known_positive"] = a["gene_id"].isin(raw_seed_set).astype(int)
        a.to_csv(target_dir / "node_filter_audit.csv", index=False)
        a[a["is_known_positive"] == 1].to_csv(target_dir / "positive_node_filter_audit.csv", index=False)
        if "kept" in a.columns:
            audit_by = a.set_index("gene_id")
            removed = [g for g in raw_seed_genes if g in audit_by.index and int(audit_by.loc[g, "kept"]) == 0]
        else:
            removed = missing_from_graph
    else:
        removed = missing_from_graph

    # Update DEG audit labels if available.
    for name in ["cold_DEG_baseline_all_genes.csv", "cold_DEG_baseline_filtered_genes.csv"]:
        src = base_dir / name
        if src.exists():
            d = pd.read_csv(src)
            if "gene_id" in d.columns:
                d["gene_id"] = d["gene_id"].apply(norm_gene_id)
                d["is_known_positive"] = d["gene_id"].isin(raw_seed_set).astype(int)
            d.to_csv(target_dir / name, index=False)

    # Condition-specific positive signal report if possible.
    deg_all_path = target_dir / "cold_DEG_baseline_all_genes.csv"
    if deg_all_path.exists():
        d = pd.read_csv(deg_all_path)
        d[d["is_known_positive"] == 1].to_csv(target_dir / "positives_signal_report.csv", index=False)

    # Summary must agree with current y.npy because Train_ColdPU validates it.
    base_summary = {}
    p = base_dir / "summary.json"
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            base_summary = json.load(f)
    summary = dict(base_summary)
    summary.update({
        "evidence_separation_condition": condition,
        "evidence_separation_label_source": str((target_dir / "seed_metadata.csv").resolve()),
        "static_features_and_graphs_reused_from": str(base_dir.resolve()),
        "node_inclusion_uses_labels": False,
        "n_known_positives_before_filter": int(len(raw_seed_set)),
        "n_known_positives_after_filter": int(y.sum()),
        "n_edgeless_known_positives_removed": int(len(removed)),
        "affected_positive_gene_ids": sorted(removed),
    })
    with (target_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Exact seed-level preprocessing audit.
    audit = seed_table.copy()
    audit["normalized_gene_id"] = audit["Pt_Gene_ID"].apply(norm_gene_id)
    audit["present_in_model_node_universe"] = audit["normalized_gene_id"].isin(node_set)
    audit["positive_in_condition_y"] = audit["normalized_gene_id"].isin(graph_seed_set)
    audit.to_csv(target_dir / "condition_seed_preprocessing_audit.csv", index=False)

    return {
        "n_raw_seeds": int(len(raw_seed_set)),
        "n_y_positive": int(y.sum()),
        "n_removed_by_label_independent_node_filter": int(len(raw_seed_set) - y.sum()),
    }


def normalize_manifest(m: pd.DataFrame) -> pd.DataFrame:
    out = m.copy()
    required = {"repeat", "split_seed", "gene_id", "partition"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    out["repeat"] = pd.to_numeric(out["repeat"], errors="raise").astype(int)
    out["split_seed"] = pd.to_numeric(out["split_seed"], errors="raise").astype(int)
    out["gene_id"] = out["gene_id"].apply(norm_gene_id)
    out["partition"] = out["partition"].astype(str).str.strip()
    return out


def validate_full_manifest(base_manifest: pd.DataFrame, full_graph_seeds: Sequence[str]) -> None:
    expected = set(full_graph_seeds)
    for rep, sub in base_manifest.groupby("repeat"):
        observed = set(sub["gene_id"])
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                "The supplied full random manifest does not match the current FULL audited y.npy after "
                f"label-independent node filtering (repeat {rep}). Missing examples={missing[:10]}, "
                f"extra examples={extra[:10]}. Regenerate the full manifest from the audited full data first."
            )


def filter_manifest_for_condition(
    base_manifest: pd.DataFrame,
    graph_seeds: Sequence[str],
    condition: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keep = set(graph_seeds)
    pieces = []
    audit_rows = []
    for rep, sub0 in base_manifest.groupby("repeat", sort=True):
        sub = sub0[sub0["gene_id"].isin(keep)].copy()
        observed = set(sub["gene_id"])
        if observed != keep:
            miss = sorted(keep - observed)
            raise ValueError(f"Condition {condition}, repeat {rep}: manifest misses {miss[:10]}")
        counts = sub["partition"].value_counts().to_dict()
        if any(int(counts.get(p, 0)) == 0 for p in ("train", "val", "test")):
            raise ValueError(
                f"Condition {condition}, repeat {rep} has an empty partition after filtering: {counts}. "
                "Do not silently re-randomize; use a newly generated manifest only if you explicitly change the design."
            )
        sub["evidence_condition"] = condition
        sub["partition_source"] = "filtered_from_full_random_manifest"
        pieces.append(sub)
        audit_rows.append({
            "condition": condition,
            "repeat": int(rep),
            "split_seed": int(sub["split_seed"].iloc[0]),
            "n_total_positive": int(len(sub)),
            "n_train": int(counts.get("train", 0)),
            "n_val": int(counts.get("val", 0)),
            "n_test": int(counts.get("test", 0)),
        })
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(audit_rows)


def cmd_prepare(args) -> None:
    base_dir = Path(args.base_data_dir).resolve()
    out_root = Path(args.out_root).resolve()
    meta_path = Path(args.seed_metadata).resolve()
    base_manifest_path = Path(args.base_manifest).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "data").mkdir(exist_ok=True)
    (out_root / "manifests").mkdir(exist_ok=True)
    (out_root / "seed_tables").mkdir(exist_ok=True)

    for req in ["X.npy", "gene_ids.txt", "edges_coexp.npy", "edges_coexp_weight.npy", "edges_ppi.npy", "edges_ppi_weight.npy"]:
        if not (base_dir / req).exists():
            raise FileNotFoundError(base_dir / req)

    meta = read_seed_metadata(meta_path, args.metadata_sheet)
    flagged = derive_evidence_flags(meta)
    flagged.to_csv(out_root / "evidence_provenance_audit.csv", index=False)

    condition_tables: Dict[str, pd.DataFrame] = {}
    raw_counts = {}
    for cond in CONDITIONS:
        t = condition_seed_table(flagged, cond)
        condition_tables[cond] = t
        raw_counts[cond] = int(t["Pt_Gene_ID"].nunique())
        t.to_csv(out_root / "seed_tables" / f"{cond}.csv", index=False)

    if not args.allow_count_mismatch:
        bad = {c: (raw_counts[c], EXPECTED_RAW_COUNTS[c]) for c in CONDITIONS if raw_counts[c] != EXPECTED_RAW_COUNTS[c]}
        if bad:
            raise ValueError(
                f"Audited seed counts differ from the expected experiment design: {bad}. "
                "Inspect evidence_provenance_audit.csv or use --allow_count_mismatch only after intentional metadata changes."
            )

    prep_rows = []
    graph_seed_sets: Dict[str, List[str]] = {}
    nodes = read_gene_ids(base_dir / "gene_ids.txt")
    node_set = set(nodes)
    for cond in CONDITIONS:
        target = out_root / "data" / cond
        counts = prepare_condition_dir(base_dir, target, condition_tables[cond], cond, args.static_mode)
        graph_seed_sets[cond] = [g for g in nodes if g in set(condition_tables[cond]["Pt_Gene_ID"].map(norm_gene_id))]
        prep_rows.append({"condition": cond, **counts})

    prep_df = pd.DataFrame(prep_rows)
    prep_df.to_csv(out_root / "condition_preprocessing_counts.csv", index=False)

    # The existing full random manifest must match the current audited FULL labels after graph filtering.
    bm = normalize_manifest(pd.read_csv(base_manifest_path))
    validate_full_manifest(bm, graph_seed_sets["full"])

    manifest_audits = []
    for cond in CONDITIONS:
        cm, ca = filter_manifest_for_condition(bm, graph_seed_sets[cond], cond)
        cm.to_csv(out_root / "manifests" / f"{cond}.csv", index=False)
        manifest_audits.append(ca)
    pd.concat(manifest_audits, ignore_index=True).to_csv(out_root / "manifest_partition_audit.csv", index=False)

    # One compact design table for reviewer response / sanity checking.
    design = prep_df.copy()
    design["n_removed_from_full_raw"] = raw_counts["full"] - design["n_raw_seeds"]
    design["definition"] = design["condition"].map({
        "full": "all audited seeds",
        "no_expression": "exclude label-defining explicit expression evidence",
        "no_go": "exclude label-defining GO annotation evidence",
        "no_expression_no_go": "exclude either explicit expression or GO annotation label evidence",
    })
    design.to_csv(out_root / "evidence_separation_design.csv", index=False)

    print("\n[OK] Evidence-separation preprocessing prepared.")
    print(design.to_string(index=False))
    print(f"\nPrepared root: {out_root}")
    print("Static X/graphs are frozen across conditions; y.npy and label-dependent reports are condition-specific.")


def load_fixed_ranking(run_dir: Path, fixed_universe: set) -> pd.DataFrame:
    p = run_dir / "ensemble_ranking.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "gene_id" not in df.columns or "mean_rank_score" not in df.columns:
        raise ValueError(f"{p} lacks gene_id/mean_rank_score")
    df["gene_id"] = df["gene_id"].apply(norm_gene_id)
    df = df[df["gene_id"].isin(fixed_universe)].copy()
    df = df.sort_values("mean_rank_score", ascending=False).reset_index(drop=True)
    df["fixed_rank"] = np.arange(1, len(df) + 1)
    return df


def rank_stats(full: pd.DataFrame, other: pd.DataFrame, condition: str, ks=(50, 100, 200)) -> Dict[str, float]:
    m = full[["gene_id", "fixed_rank"]].merge(
        other[["gene_id", "fixed_rank"]], on="gene_id", suffixes=("_full", "_condition")
    )
    if len(m) < 2:
        raise ValueError(f"Too few common genes for ranking comparison: {condition}")
    if HAVE_SCIPY:
        rho = float(spearmanr(m["fixed_rank_full"], m["fixed_rank_condition"]).statistic)
        tau = float(kendalltau(m["fixed_rank_full"], m["fixed_rank_condition"]).statistic)
    else:
        rho = float(pd.Series(m["fixed_rank_full"]).corr(pd.Series(m["fixed_rank_condition"]), method="spearman"))
        tau = np.nan
    row = {
        "condition": condition,
        "n_fixed_candidates": int(len(m)),
        "spearman_vs_full": rho,
        "kendall_tau_vs_full": tau,
    }
    for k0 in ks:
        k = min(k0, len(m))
        a = set(m.nsmallest(k, "fixed_rank_full")["gene_id"])
        b = set(m.nsmallest(k, "fixed_rank_condition")["gene_id"])
        inter = len(a & b)
        row[f"top{k0}_overlap_n"] = inter
        row[f"top{k0}_overlap_fraction"] = inter / k if k else np.nan
        row[f"top{k0}_jaccard"] = inter / len(a | b) if (a | b) else np.nan
    top200 = m.nsmallest(min(200, len(m)), "fixed_rank_full")
    shift = np.abs(top200["fixed_rank_full"].to_numpy() - top200["fixed_rank_condition"].to_numpy())
    row["median_abs_rank_shift_full_top200"] = float(np.median(shift))
    row["mean_abs_rank_shift_full_top200"] = float(np.mean(shift))
    return row


def cmd_summarize(args) -> None:
    prepared = Path(args.prepared_root).resolve()
    runs = Path(args.runs_root).resolve()
    out = Path(args.out_dir).resolve() if args.out_dir else prepared / "reviewer_results"
    out.mkdir(parents=True, exist_ok=True)

    nodes = read_gene_ids(prepared / "data" / "full" / "gene_ids.txt")
    full_y = np.load(prepared / "data" / "full" / "y.npy").astype(int)
    full_seed_set = set(np.asarray(nodes, dtype=object)[full_y == 1].tolist())
    fixed_universe = set(nodes) - full_seed_set

    rankings = {}
    for cond in CONDITIONS:
        rankings[cond] = load_fixed_ranking(runs / cond, fixed_universe)
        rankings[cond].to_csv(out / f"fixed_universe_ranking__{cond}.csv", index=False)

    stability_rows = []
    for cond in ("no_expression", "no_go", "no_expression_no_go"):
        stability_rows.append(rank_stats(rankings["full"], rankings[cond], cond))
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(out / "candidate_rank_stability_vs_full.csv", index=False)

    # Collect metric summaries descriptively; test sets differ across conditions, so no paired-performance claim.
    metric_rows = []
    for cond in CONDITIONS:
        p = runs / cond / "summary_metrics_95CI.csv"
        if p.exists():
            x = pd.read_csv(p)
            x.insert(0, "condition", cond)
            metric_rows.append(x)
    if metric_rows:
        pd.concat(metric_rows, ignore_index=True).to_csv(out / "condition_metric_summary.csv", index=False)

    counts = pd.read_csv(prepared / "condition_preprocessing_counts.csv")
    reviewer = counts.merge(stability, on="condition", how="left")
    reviewer.to_csv(out / "reviewer_evidence_separation_summary.csv", index=False)

    # Candidate rank matrix across conditions on the same fixed universe.
    mat = None
    for cond in CONDITIONS:
        x = rankings[cond][["gene_id", "fixed_rank", "mean_rank_score"]].rename(
            columns={"fixed_rank": f"rank__{cond}", "mean_rank_score": f"score__{cond}"}
        )
        mat = x if mat is None else mat.merge(x, on="gene_id", how="outer")
    mat.to_csv(out / "candidate_rank_matrix.csv", index=False)

    print("\n[OK] Reviewer-facing evidence-separation summaries written to:")
    print(out)
    print(stability.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare and summarize PoplarCold-PUGAT evidence-separation experiments")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("prepare", help="prepare condition-specific y/data/seed tables/manifests")
    a.add_argument("--base_data_dir", required=True)
    a.add_argument("--seed_metadata", required=True)
    a.add_argument("--metadata_sheet", default="Supplementary_Data_1_Revised")
    a.add_argument("--base_manifest", required=True, help="full audited RANDOM split_manifest.csv")
    a.add_argument("--out_root", required=True)
    a.add_argument("--static_mode", choices=["copy", "hardlink", "symlink"], default="hardlink")
    a.add_argument("--allow_count_mismatch", action="store_true")
    a.set_defaults(func=cmd_prepare)

    s = sub.add_parser("summarize", help="summarize rankings after all four training runs")
    s.add_argument("--prepared_root", required=True)
    s.add_argument("--runs_root", required=True, help="contains full/, no_expression/, no_go/, no_expression_no_go/")
    s.add_argument("--out_dir", default="")
    s.set_defaults(func=cmd_summarize)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

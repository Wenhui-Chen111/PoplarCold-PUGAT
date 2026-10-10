#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reviewer-requested controls for the external DEG-overlap analysis.

Compares the model top set with: (1) an expression-only ranking, (2) uniform
random sets, and (3) random sets matched for expression and network degree.
Only the compact comparison table is written.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def norm_gene_id(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().split()[0].strip('"')
    if not s or s.lower() == "nan":
        return None
    for prefix in ("POPTR_", "POPTR.", "Potri_"):
        if s.startswith(prefix):
            s = "Potri." + s[len(prefix):]
            break
    if s.startswith("Potri."):
        return re.sub(r"v\d+$", "", s)
    m = re.search(r"(Potri\.(?:\d+G\d+|T\d+))", s)
    return m.group(1) if m else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="External-validation controls")
    p.add_argument("--ranking", required=True, help="Full ensemble_ranking.csv")
    p.add_argument("--features", required=True, help="features_raw_table.csv")
    p.add_argument("--external_data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_n", type=int, default=200)
    p.add_argument("--n_perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--readcount_file", default="Pt_read_count data.xlsx")
    p.add_argument("--readcount_sheet", default="Pt.stringtie.gene.TMM.EXPR")
    p.add_argument("--expression_score", default="max_abs_cold_fc")
    p.add_argument("--matched_covariates", nargs="+", default=[
        "max_abs_cold_fc", "baseline_log_expr", "log_coexp_degree", "log_ppi_degree"
    ])
    p.add_argument("--propensity_bins", type=int, default=10)
    return p.parse_args()


def load_ranking(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    gc = "gene_id" if "gene_id" in d.columns else d.columns[0]
    d = d.copy()
    d["gene_id"] = d[gc].map(norm_gene_id)
    d = d[d["gene_id"].notna()].drop_duplicates("gene_id")
    if "rank" in d.columns:
        d = d.sort_values("rank")
    elif "mean_rank_score" in d.columns:
        d = d.sort_values("mean_rank_score", ascending=False)
    elif "rank_score" in d.columns:
        d = d.sort_values("rank_score", ascending=False)
    else:
        raise ValueError("Ranking file needs rank, mean_rank_score, or rank_score")
    return d.reset_index(drop=True)


def find_sheet(path: Path, tp: int) -> str | int:
    xl = pd.ExcelFile(path)
    for s in xl.sheet_names:
        if f"{tp}h" in s:
            return s
    return xl.sheet_names[0]


def load_external(data_dir: Path, readcount_file: str, readcount_sheet: str):
    rc = pd.read_excel(data_dir / readcount_file, sheet_name=readcount_sheet, usecols=[0])
    ext = {norm_gene_id(x) for x in rc.iloc[:, 0]}
    ext.discard(None)

    p168 = data_dir / "Pt_L168h_DEGs.xlsx"
    if not p168.exists():
        p168 = data_dir / "Pt_L68h_DEGs.xlsx"
    paths = {2: data_dir / "Pt_L2h_DEGs.xlsx", 24: data_dir / "Pt_L24h_DEGs.xlsx", 168: p168}
    deg = {}
    for tp, path in paths.items():
        d = pd.read_excel(path, sheet_name=find_sheet(path, tp), usecols=[0])
        s = {norm_gene_id(x) for x in d.iloc[:, 0]}
        s.discard(None)
        deg[tp] = s
    return ext, deg


def overlap(genes: set[str], deg: set[str], bg: set[str]) -> tuple[int, int, float, float, float]:
    genes, deg = genes & bg, deg & bg
    K, k = len(genes), len(genes & deg)
    bg_frac = len(deg) / len(bg)
    frac = k / K
    return K, k, frac, bg_frac, frac / bg_frac


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ranking = load_ranking(args.ranking)
    feat = pd.read_csv(args.features)
    if "gene_id" not in feat.columns:
        raise ValueError("features_raw_table.csv must contain gene_id")
    feat = feat.copy()
    feat["gene_id"] = feat["gene_id"].map(norm_gene_id)
    feat = feat[feat["gene_id"].notna()].drop_duplicates("gene_id")

    ext, per_tp = load_external(Path(args.external_data_dir), args.readcount_file, args.readcount_sheet)
    union_all = per_tp[2] | per_tp[24] | per_tp[168]

    needed = list(dict.fromkeys([args.expression_score] + args.matched_covariates))
    missing = [c for c in needed if c not in feat.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    for c in needed:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")

    # Use one common complete-case universe for all four comparisons.
    df = ranking[["gene_id"]].merge(feat[["gene_id"] + needed], on="gene_id", how="inner", validate="one_to_one")
    df = df[df["gene_id"].isin(ext)].dropna(subset=needed).reset_index(drop=True)
    bg = set(df["gene_id"])

    requested_top = set(ranking.head(args.top_n)["gene_id"])
    model_top = requested_top & bg
    K = len(model_top)
    if K == 0:
        raise ValueError("No model top genes map to the common control-analysis universe")

    expr_top = set(df.sort_values(args.expression_score, ascending=False).head(K)["gene_id"])

    X = df[args.matched_covariates].to_numpy(float)
    y = df["gene_id"].isin(model_top).astype(int).to_numpy()
    clf = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=args.seed
    ))
    clf.fit(X, y)
    df["propensity"] = clf.predict_proba(X)[:, 1]
    q = min(args.propensity_bins, max(2, len(df) // 50))
    df["prop_bin"] = pd.qcut(df["propensity"], q=q, duplicates="drop")
    top_bin_counts = df[df["gene_id"].isin(model_top)].groupby("prop_bin", observed=True).size()

    def draw_matched() -> set[str]:
        picked = []
        for bin_label, n_take in top_bin_counts.items():
            pool = df.loc[df["prop_bin"] == bin_label, "gene_id"].to_numpy(object)
            picked.extend(rng.choice(pool, size=int(n_take), replace=False).tolist())
        return set(picked)

    bg_arr = np.array(sorted(bg), dtype=object)
    uniform = np.empty(args.n_perm)
    matched = np.empty(args.n_perm)
    deg = union_all & bg
    for i in range(args.n_perm):
        u = set(rng.choice(bg_arr, size=K, replace=False).tolist())
        m = draw_matched()
        uniform[i] = len(u & deg) / K
        matched[i] = len(m & deg) / K

    model = overlap(model_top, union_all, bg)
    expr = overlap(expr_top, union_all, bg)
    bg_frac = model[3]
    rows = [
        {"Comparison": "Model", "K": model[0], "overlap_k": model[1],
         "Fraction": round(model[2], 6), "Background fraction": round(bg_frac, 6), "Fold": round(model[4], 5)},
        {"Comparison": f"Expression-only: {args.expression_score}", "K": expr[0], "overlap_k": expr[1],
         "Fraction": round(expr[2], 6), "Background fraction": round(bg_frac, 6), "Fold": round(expr[4], 5)},
        {"Comparison": "Uniform random mean", "K": K, "overlap_k": "—",
         "Fraction": round(float(uniform.mean()), 6), "Background fraction": round(bg_frac, 6),
         "Fold": round(float(uniform.mean() / bg_frac), 5)},
        {"Comparison": "Expression/degree-matched random mean", "K": K, "overlap_k": "—",
         "Fraction": round(float(matched.mean()), 6), "Background fraction": round(bg_frac, 6),
         "Fold": round(float(matched.mean() / bg_frac), 5)},
    ]

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "Table4_external_validation_controls.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

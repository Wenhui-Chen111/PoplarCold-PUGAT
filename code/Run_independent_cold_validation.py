#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External cold-expression validation using the published DEG calls of Wang et al. (2023).

No differential-expression analysis is rerun here.  The TMM workbook is used only
for gene mapping/background definition; the deposited DEG tables are used as published.
Outputs are intentionally limited to the three reviewer-facing tables.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


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
    p = argparse.ArgumentParser(description="Independent cold validation with published DEG calls")
    p.add_argument("--ranking", required=True, help="Full ensemble_ranking.csv (known seeds already excluded)")
    p.add_argument("--data_dir", required=True, help="Directory containing the external xlsx files")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_n", type=int, default=200)
    p.add_argument("--n_perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--readcount_file", default="Pt_read_count data.xlsx")
    p.add_argument("--readcount_sheet", default="Pt.stringtie.gene.TMM.EXPR")
    p.add_argument("--deg_2h", default="Pt_L2h_DEGs.xlsx")
    p.add_argument("--deg_24h", default="Pt_L24h_DEGs.xlsx")
    p.add_argument("--deg_168h", default=None,
                   help="Default: Pt_L168h_DEGs.xlsx, falling back to Pt_L68h_DEGs.xlsx")
    return p.parse_args()


def load_ranking(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    gene_col = next((c for c in ("gene_id", "gene", "Gene_ID", "Gene_id", "id") if c in d.columns), d.columns[0])
    d = d.copy()
    d["gene_id"] = d[gene_col].map(norm_gene_id)
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


def load_external_universe(path: Path, sheet: str) -> set[str]:
    d = pd.read_excel(path, sheet_name=sheet, usecols=[0])
    out = {norm_gene_id(x) for x in d.iloc[:, 0]}
    out.discard(None)
    return out


def load_published_deg(path: Path, tp: int) -> pd.DataFrame:
    d = pd.read_excel(path, sheet_name=find_sheet(path, tp))
    d = d.copy()
    d["gene_id"] = d.iloc[:, 0].map(norm_gene_id)
    d = d[d["gene_id"].notna()].drop_duplicates("gene_id")

    # These files are the source study's published DEG calls; do not re-test rows here.
    if "Programs" in d.columns:
        programs = pd.to_numeric(d["Programs"], errors="coerce")
        if (programs.dropna() < 2).any():
            raise ValueError(f"Unexpected Programs < 2 in published DEG table: {path}")

    fc_col = "edgeR-log2FC" if "edgeR-log2FC" in d.columns else next(
        (c for c in d.columns if "log2fc" in str(c).lower()), None
    )
    status_col = "Status" if "Status" in d.columns else None
    out = pd.DataFrame(index=d["gene_id"])
    out["log2FC"] = pd.to_numeric(d[fc_col], errors="coerce").to_numpy() if fc_col else np.nan
    out["Status"] = d[status_col].astype(str).to_numpy() if status_col else ""
    return out


def fisher_stats(candidates: set[str], bg: set[str], deg: set[str]) -> dict:
    c, d = candidates & bg, deg & bg
    M, K, n, k = len(bg), len(c), len(d), len(c & d)
    table = [[k, K - k], [n - k, M - K - n + k]]
    _, p = fisher_exact(table, alternative="greater")
    cand_frac, bg_frac = k / K, n / M
    return dict(k=k, K=K, n=n, M=M, cand_frac=cand_frac, bg_frac=bg_frac,
                fold=cand_frac / bg_frac, p=p)


def direction(status: str, fc) -> str:
    s = str(status).upper()
    if "UP" in s:
        return "UP"
    if "DOWN" in s:
        return "DOWN"
    try:
        return "UP" if float(fc) > 0 else ("DOWN" if float(fc) < 0 else "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    rng = np.random.default_rng(args.seed)

    ranking = load_ranking(args.ranking)
    top = ranking.head(args.top_n)
    external = load_external_universe(data_dir / args.readcount_file, args.readcount_sheet)
    eligible_bg = set(ranking["gene_id"]) & external
    mapped_top = set(top["gene_id"]) & eligible_bg
    if not mapped_top:
        raise ValueError("No top-ranked genes map to the eligible external background")

    deg168 = args.deg_168h
    if deg168 is None:
        deg168 = "Pt_L168h_DEGs.xlsx" if (data_dir / "Pt_L168h_DEGs.xlsx").exists() else "Pt_L68h_DEGs.xlsx"
    paths = {2: data_dir / args.deg_2h, 24: data_dir / args.deg_24h, 168: data_dir / deg168}
    per_tp = {tp: load_published_deg(path, tp) for tp, path in paths.items()}
    deg_sets = {tp: set(df.index) for tp, df in per_tp.items()}
    union_all = deg_sets[2] | deg_sets[24] | deg_sets[168]
    union_24_168 = deg_sets[24] | deg_sets[168]

    # Table 1: enrichment summary.
    tests = [
        ("Union 2+24+168 h", union_all),
        ("Union 24+168 h sensitivity", union_24_168),
        ("2 h", deg_sets[2]),
        ("24 h", deg_sets[24]),
        ("168 h", deg_sets[168]),
    ]
    stats = [(label, fisher_stats(mapped_top, eligible_bg, d)) for label, d in tests]
    qvals = multipletests([s[1]["p"] for s in stats], method="fdr_bh")[1]
    table1 = []
    for (label, s), q in zip(stats, qvals):
        table1.append({
            "Set": label,
            "overlap k/K": f'{s["k"]}/{s["K"]}',
            "Candidate %": f'{100*s["cand_frac"]:.2f}%',
            "Background n/M": f'{s["n"]}/{s["M"]}',
            "Background %": f'{100*s["bg_frac"]:.2f}%',
            "Fold enrichment": round(s["fold"], 4),
            "Fisher P": round(s["p"], 6),
            "BH q": round(float(q), 6),
        })
    pd.DataFrame(table1).to_csv(outdir / "Table1_external_enrichment.csv", index=False)

    # Table 2: uniform permutation test for the full union.
    pool = np.array(sorted(eligible_bg), dtype=object)
    deg = union_all & eligible_bg
    K = len(mapped_top)
    obs_k = len(mapped_top & deg)
    obs_frac = obs_k / K
    perm = np.empty(args.n_perm, dtype=float)
    for i in range(args.n_perm):
        draw = rng.choice(pool, size=K, replace=False)
        perm[i] = sum(g in deg for g in draw) / K
    p_emp = (int(np.sum(perm >= obs_frac)) + 1) / (args.n_perm + 1)
    pd.DataFrame([{
        "set": "union_2_24_168h",
        "observed_overlap": obs_k,
        "candidate_mapped_K": K,
        "observed_fraction": round(obs_frac, 6),
        "random_mean_fraction": round(float(perm.mean()), 6),
        "random_sd_fraction": round(float(perm.std(ddof=1)), 6),
        "empirical_p": round(float(p_emp), 6),
    }]).to_csv(outdir / "Table2_uniform_permutation.csv", index=False)

    # Table 3: direction at the earliest published DEG time point.
    rows = []
    for g in sorted(mapped_top & union_all):
        for tp in (2, 24, 168):
            if g in per_tp[tp].index:
                rows.append((tp, direction(per_tp[tp].at[g, "Status"], per_tp[tp].at[g, "log2FC"])))
                break
    table3 = (pd.DataFrame(rows, columns=["earliest_responsive_time_h", "direction_at_earliest_response"])
              .groupby(["earliest_responsive_time_h", "direction_at_earliest_response"], sort=True)
              .size().reset_index(name="n"))
    table3.to_csv(outdir / "Table3_earliest_response.csv", index=False)

    print(pd.DataFrame(table1).to_string(index=False))
    print("\n", pd.read_csv(outdir / "Table2_uniform_permutation.csv").to_string(index=False), sep="")
    print("\n", table3.to_string(index=False), sep="")


if __name__ == "__main__":
    main()

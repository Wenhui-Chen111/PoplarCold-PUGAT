#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrichment analysis for PoplarCold-PUGAT.

Design goals
------------
1. Use the FINAL ensemble_ranking.csv directly. The revised training code already
   removes all known seed positives from this file.
2. Define an explicit, analysis-specific eligible gene universe.
3. Report the complete 2x2 table, candidate/background counts, effect sizes,
   raw Fisher P values, and BH-adjusted q values.
4. Use the hit-only BH procedure used by this analysis:
   after applying the prespecified minimum term/family-size filter, remove
   hypotheses with zero top-200 hits (k = 0), then perform BH correction only
   across hypotheses with at least one candidate hit.
5. For GO seed-construction terms, compute an audit P value but exclude the
   prespecified terms from the PRIMARY GO hypothesis family. Also provide q_all,
   obtained with those terms included, so the reviewer can verify both treatments.
6. Use q < 0.05 as the inferential reporting threshold. Values 0.05 <= q < 0.10
   are reported only as descriptive trends.

Suggested command
-----------------
python Run_enrichment.py \
  --ranking ensemble_ranking.csv \
  --go data/Ptr_GO_annotation \
  --kegg-gene-ko data/Ptrichocarpa.annotation_info.txt \
  --keg data/pop00001.keg \
  --tf data/Ptr_TF_list.txt \
  --go-seed-terms GO:0009631 GO:0050826 GO:0009409 GO:0071497 \
  --outdir results/enrichment

IMPORTANT: --go-seed-terms must contain the COMPLETE set of GO terms that were
actually used directly in the REVISED seed-construction procedure. The two GO IDs
above are defaults only because they are named in the current manuscript text.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

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
    # Keep already-clean nonstandard IDs rather than silently dropping them.
    return s


def read_table_flexible(path: Path) -> pd.DataFrame:
    """Read a text table, preferring tab separation but falling back to sniffing."""
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, comment=None, low_memory=False)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=None, engine="python", dtype=str)


def resolve_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        hit = normalized.get(c.strip().lower())
        if hit is not None:
            return hit
    if required:
        raise ValueError(
            f"None of the required columns {list(candidates)} were found. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def write_gene_list(path: Path, genes: Iterable[str]) -> None:
    vals = sorted(set(genes))
    path.write_text("\n".join(vals) + ("\n" if vals else ""), encoding="utf-8")


def q_status(q: float) -> str:
    if pd.isna(q):
        return "not_in_primary_family"
    if q < 0.05:
        return "significant_q_lt_0.05"
    if q < 0.10:
        return "descriptive_0.05_le_q_lt_0.10"
    return "not_significant_q_ge_0.10"


def fisher_stats(k: int, n_candidate: int, k_bg: int, n_bg: int) -> Dict[str, float]:
    """
    Candidate set is a subset of the eligible background.

    2x2 table:
                      in term     not in term
      candidate          a             b
      non-candidate      c             d

    where a=k, b=n_candidate-k, c=k_bg-k,
          d=n_bg-n_candidate-k_bg+k.
    """
    a = int(k)
    b = int(n_candidate - k)
    c = int(k_bg - k)
    d = int(n_bg - n_candidate - k_bg + k)
    if min(a, b, c, d) < 0:
        raise ValueError(
            f"Invalid contingency table: a={a}, b={b}, c={c}, d={d}; "
            f"k={k}, N={n_candidate}, K={k_bg}, M={n_bg}"
        )

    odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
    fold = (a / n_candidate) / (k_bg / n_bg) if n_candidate > 0 and k_bg > 0 and n_bg > 0 else np.nan

    # Haldane-Anscombe corrected OR and approximate 95% CI; finite even with zero cells.
    ah, bh, ch, dh = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ha = (ah * dh) / (bh * ch)
    se_log_or = math.sqrt(1.0 / ah + 1.0 / bh + 1.0 / ch + 1.0 / dh)
    lo = math.exp(math.log(or_ha) - 1.96 * se_log_or)
    hi = math.exp(math.log(or_ha) + 1.96 * se_log_or)

    return {
        "a_candidate_in_term": a,
        "b_candidate_not_in_term": b,
        "c_noncandidate_in_term": c,
        "d_noncandidate_not_in_term": d,
        "candidate_fraction": a / n_candidate if n_candidate else np.nan,
        "background_fraction": k_bg / n_bg if n_bg else np.nan,
        "fold_enrichment": fold,
        "odds_ratio_fisher": float(odds_ratio),
        "odds_ratio_haldane": float(or_ha),
        "or95ci_low_haldane": float(lo),
        "or95ci_high_haldane": float(hi),
        "p_value": float(p_value),
    }


def bh_adjust(p: pd.Series) -> pd.Series:
    if len(p) == 0:
        return pd.Series(dtype=float, index=p.index)
    vals = pd.to_numeric(p, errors="raise").to_numpy(dtype=float)
    return pd.Series(multipletests(vals, method="fdr_bh")[1], index=p.index, dtype=float)


def load_ranking(path: Path, top_n: int) -> Tuple[pd.DataFrame, Set[str], List[str], Set[str]]:
    df = pd.read_csv(path)
    gene_col = resolve_col(df, ["gene_id", "Gene_ID", "gene", "Gene"])
    df = df.copy()
    df[gene_col] = df[gene_col].apply(norm_gene_id)
    df = df[df[gene_col].notna()].copy()
    df = df.rename(columns={gene_col: "gene_id"})

    if df["gene_id"].duplicated().any():
        dup = df.loc[df["gene_id"].duplicated(keep=False), "gene_id"].head(20).tolist()
        raise ValueError(f"Duplicate gene IDs in ranking after normalization: {dup}")

    # Defensive handling: Train_ColdPU already removes known seeds, but fail-safe here.
    if "is_known_seed" in df.columns:
        x = df["is_known_seed"]
        if x.dtype == bool:
            known = x
        else:
            known = x.astype(str).str.lower().isin({"true", "1", "yes", "y"})
        if known.any():
            print(f"[warning] ranking contained {int(known.sum())} known seeds; removing them defensively")
            df = df.loc[~known].copy()

    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        if df["rank"].isna().any():
            raise ValueError("Ranking column contains non-numeric values")
        df = df.sort_values("rank", ascending=True)
    elif "mean_rank_score" in df.columns:
        df = df.sort_values("mean_rank_score", ascending=False)
    elif "rank_score" in df.columns:
        df = df.sort_values("rank_score", ascending=False)
    else:
        raise ValueError("Ranking needs 'rank', 'mean_rank_score', or 'rank_score'.")

    df = df.reset_index(drop=True)
    if len(df) < top_n:
        raise ValueError(f"Ranking contains only {len(df)} genes, fewer than TOP_N={top_n}")

    base_universe = set(df["gene_id"])
    top_list = df.loc[: top_n - 1, "gene_id"].tolist()
    top_set = set(top_list)
    if len(top_set) != top_n:
        raise ValueError("Top-N ranking contains duplicated normalized gene IDs")
    return df, base_universe, top_list, top_set


# -----------------------------------------------------------------------------
# GO-BP
# -----------------------------------------------------------------------------

def load_go(path: Path) -> pd.DataFrame:
    df = read_table_flexible(path)
    gene_col = resolve_col(df, ["Gene_id", "Gene_ID", "gene_id", "gene"])
    goid_col = resolve_col(df, ["GO_ID", "GO ID", "go_id", "go"])
    term_col = resolve_col(df, ["GO_term", "GO term", "term", "Description", "name"])
    aspect_col = resolve_col(df, ["Aspect", "aspect", "namespace"], required=False)

    out = pd.DataFrame({
        "gene_id": df[gene_col].apply(norm_gene_id),
        "GO_ID": df[goid_col].astype(str).str.strip(),
        "GO_term": df[term_col].astype(str).str.strip(),
    })
    if aspect_col is not None:
        aspect = df[aspect_col].astype(str).str.strip().str.lower()
        bp_mask = aspect.isin({"p", "bp", "biological_process", "biological process"})
        out = out.loc[bp_mask].copy()
    out = out.dropna(subset=["gene_id", "GO_ID", "GO_term"])
    out = out[out["GO_ID"].str.match(r"^GO:\d+$", na=False)]
    return out.drop_duplicates(["gene_id", "GO_ID", "GO_term"]).reset_index(drop=True)


def run_go(
    go: pd.DataFrame,
    base_universe: Set[str],
    top_set: Set[str],
    excluded_seed_terms: Set[str],
    min_term_size: int,
    outdir: Path,
) -> Dict[str, object]:
    go = go[go["gene_id"].isin(base_universe)].copy()
    eligible = set(go["gene_id"])
    top_eligible = top_set & eligible
    M, N = len(eligible), len(top_eligible)
    if M == 0 or N == 0:
        raise ValueError(f"GO-BP eligible universe is empty or top candidates do not map: M={M}, N={N}")

    write_gene_list(outdir / "eligible_universe_GO_BP.txt", eligible)
    write_gene_list(outdir / "top200_eligible_GO_BP.txt", top_eligible)

    # One hypothesis per GO_ID. Never group by GO term name.
    rows: List[Dict[str, object]] = []
    tested_ids: Set[str] = set()
    term_name_by_id = (
        go.groupby("GO_ID")["GO_term"]
        .agg(lambda x: x.value_counts().index[0] if len(x) else "")
        .to_dict()
    )
    genes_by_id = go.groupby("GO_ID")["gene_id"].apply(lambda s: set(s) & eligible).to_dict()

    for go_id in sorted(genes_by_id):
        genes = genes_by_id[go_id]
        K = len(genes)
        if K < min_term_size:
            continue
        tested_ids.add(go_id)
        hits = genes & top_eligible
        stat = fisher_stats(len(hits), N, K, M)
        rows.append({
            "GO_ID": go_id,
            "GO_term": term_name_by_id.get(go_id, ""),
            "excluded_seed_term": go_id in excluded_seed_terms,
            "candidate_hits": len(hits),
            "candidate_eligible_n": N,
            "background_members": K,
            "background_eligible_n": M,
            **stat,
            "hit_genes": ";".join(sorted(hits)),
        })

    res_all = pd.DataFrame(rows)
    if res_all.empty:
        raise ValueError("No GO-BP terms passed the minimum term-size filter")

    # ------------------------------------------------------------------
    # LEGACY / HIT-ONLY BH:
    # First remove k=0 terms, then perform BH only among terms with >=1
    # top-candidate hit.  This reproduces the behavior of the original code.
    # ------------------------------------------------------------------
    zero_hit = res_all[res_all["candidate_hits"] == 0].copy()
    zero_hit["BH_treatment"] = "Excluded before BH because candidate_hits == 0"
    zero_hit.to_csv(outdir / "GO_BP_zero_hit_excluded_from_BH.csv", index=False)

    res = res_all[res_all["candidate_hits"] > 0].copy()
    if res.empty:
        raise ValueError("No GO-BP term has >=1 top-candidate hit after filtering")

    # Audit correction including seed-construction terms, but only among hit terms.
    res["q_value_all_terms"] = bh_adjust(res["p_value"])

    # PRIMARY correction: prespecified seed-construction GO terms are excluded
    # from the primary family, and k=0 terms have already been removed.
    res["q_value"] = np.nan
    primary = ~res["excluded_seed_term"]
    res.loc[primary, "q_value"] = bh_adjust(res.loc[primary, "p_value"])
    res["status"] = res["q_value"].apply(q_status)
    res["tested_in_primary_BH_family"] = primary
    res["n_hypotheses_eligible_before_hit_filter"] = len(res_all)
    res["n_hypotheses_all_hit_terms"] = len(res)
    res["n_hypotheses_primary"] = int(primary.sum())

    # Full membership table for exact verification.
    membership = go[go["GO_ID"].isin(tested_ids) & go["gene_id"].isin(eligible)][
        ["GO_ID", "GO_term", "gene_id"]
    ].drop_duplicates()
    membership["is_top200"] = membership["gene_id"].isin(top_set)
    membership["is_top200_and_GO_eligible"] = membership["gene_id"].isin(top_eligible)
    membership["excluded_seed_term"] = membership["GO_ID"].isin(excluded_seed_terms)
    membership.to_csv(outdir / "GO_BP_term_membership_long.csv", index=False)

    # Report requested excluded terms even if absent from current annotations/testing.
    excluded_rows = []
    indexed = res.set_index("GO_ID", drop=False)
    for go_id in sorted(excluded_seed_terms):
        if go_id in indexed.index:
            r = indexed.loc[go_id]
            excluded_rows.append({
                "GO_ID": go_id,
                "GO_term": r["GO_term"],
                "present_in_testable_annotation": True,
                "background_members": int(r["background_members"]),
                "candidate_hits": int(r["candidate_hits"]),
                "raw_p_for_audit": float(r["p_value"]),
                "q_value_all_terms_for_audit": float(r["q_value_all_terms"]),
                "q_value_primary": np.nan,
                "multiple_testing_treatment": "Excluded a priori from primary BH family; retained only for audit",
            })
        else:
            excluded_rows.append({
                "GO_ID": go_id,
                "GO_term": term_name_by_id.get(go_id, ""),
                "present_in_testable_annotation": False,
                "background_members": np.nan,
                "candidate_hits": np.nan,
                "raw_p_for_audit": np.nan,
                "q_value_all_terms_for_audit": np.nan,
                "q_value_primary": np.nan,
                "multiple_testing_treatment": "Requested exclusion term not present among testable GO-BP terms",
            })
    pd.DataFrame(excluded_rows).to_csv(outdir / "GO_seed_construction_terms_excluded.csv", index=False)

    res = res.sort_values(["excluded_seed_term", "q_value", "p_value"], na_position="last")
    res.to_csv(outdir / "Table_S3_GO_BP_all_tested.csv", index=False)
    reported = res[(~res["excluded_seed_term"]) & (res["q_value"] < 0.10)].copy()
    reported.to_csv(outdir / "Table_S3_GO_BP_reported_q_lt_0.10.csv", index=False)

    return {
        "analysis": "GO-BP",
        "ranking_universe_n": len(base_universe),
        "top_candidates_requested_n": len(top_set),
        "eligible_background_n": M,
        "eligible_top_candidates_n": N,
        "mapping_success_fraction_top": N / len(top_set),
        "min_term_size": min_term_size,
        "hypotheses_all_n": len(res_all),
        "hypotheses_primary_BH_n": int(primary.sum()),
        "hypotheses_zero_hit_excluded_n": int((res_all["candidate_hits"] == 0).sum()),
        "hypotheses_hit_terms_n": len(res),
        "excluded_seed_terms_requested_n": len(excluded_seed_terms),
        "significant_q_lt_0.05_n": int(((res["q_value"] < 0.05) & ~res["excluded_seed_term"]).sum()),
        "descriptive_0.05_le_q_lt_0.10_n": int((((res["q_value"] >= 0.05) & (res["q_value"] < 0.10)) & ~res["excluded_seed_term"]).sum()),
        "background_definition": "genes in ensemble_ranking.csv with >=1 GO-BP annotation",
        "candidate_definition": "top-ranked 200 genes intersected with GO-BP eligible background",
        "BH_definition": "legacy hit-only BH: after min-size filtering, remove candidate_hits==0 terms; primary BH across remaining non-seed GO-BP hit terms",
    }


# -----------------------------------------------------------------------------
# KEGG
# -----------------------------------------------------------------------------

def parse_keg(path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    ko2path: Dict[str, Set[str]] = {}
    path_name: Dict[str, str] = {}
    current: Optional[str] = None
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("C"):
                m = re.search(r"\[PATH:(pop\d+)\]", line)
                current = m.group(1) if m else None
                if current:
                    name = re.sub(r"^C\s*\d+\s*", "", line).split("[PATH:")[0].strip()
                    path_name[current] = name
            elif line.startswith("D") and current:
                # Keep all KO IDs on the line, not only the first one.
                for ko in re.findall(r"K\d{5}", line):
                    ko2path.setdefault(ko, set()).add(current)
    return ko2path, path_name


def load_gene2ko(path: Path) -> Dict[str, Set[str]]:
    df = read_table_flexible(path)
    id_col = resolve_col(df, ["locusName", "locus_name", "Gene_ID", "gene_id", "gene", "Name"], required=False)
    if id_col is None:
        if len(df.columns) < 2:
            raise ValueError("Cannot infer gene-ID column in KEGG gene-to-KO annotation table")
        id_col = df.columns[1]
    ko_col = resolve_col(df, ["KO", "ko", "KEGG_KO", "KEGG Orthology"], required=True)

    g2k: Dict[str, Set[str]] = {}
    for _, r in df[[id_col, ko_col]].iterrows():
        gid = norm_gene_id(r[id_col])
        if gid is None:
            continue
        kos = set(re.findall(r"K\d{5}", str(r[ko_col])))
        if kos:
            g2k.setdefault(gid, set()).update(kos)
    return g2k


def run_kegg(
    g2k: Dict[str, Set[str]],
    ko2path: Dict[str, Set[str]],
    path_name: Dict[str, str],
    base_universe: Set[str],
    top_set: Set[str],
    min_path_size: int,
    outdir: Path,
) -> Dict[str, object]:
    gene2path: Dict[str, Set[str]] = {}
    for g in base_universe:
        pset: Set[str] = set()
        for ko in g2k.get(g, set()):
            pset.update(ko2path.get(ko, set()))
        if pset:
            gene2path[g] = pset

    # "Valid KEGG annotation" is defined as membership in >=1 parsed P. trichocarpa pathway.
    eligible = set(gene2path)
    top_eligible = top_set & eligible
    M, N = len(eligible), len(top_eligible)
    if M == 0 or N == 0:
        raise ValueError(f"KEGG eligible universe is empty or top candidates do not map: M={M}, N={N}")

    write_gene_list(outdir / "eligible_universe_KEGG.txt", eligible)
    write_gene_list(outdir / "top200_eligible_KEGG.txt", top_eligible)

    path2genes: Dict[str, Set[str]] = {}
    for g, paths in gene2path.items():
        for p in paths:
            path2genes.setdefault(p, set()).add(g)

    rows: List[Dict[str, object]] = []
    tested_paths: Set[str] = set()
    for pid in sorted(path2genes):
        genes = path2genes[pid]
        K = len(genes)
        if K < min_path_size:
            continue
        tested_paths.add(pid)
        hits = genes & top_eligible
        stat = fisher_stats(len(hits), N, K, M)
        rows.append({
            "pathway_id": pid,
            "Description": path_name.get(pid, pid),
            "candidate_hits": len(hits),
            "candidate_eligible_n": N,
            "background_members": K,
            "background_eligible_n": M,
            **stat,
            "hit_genes": ";".join(sorted(hits)),
        })

    res_all = pd.DataFrame(rows)
    if res_all.empty:
        raise ValueError("No KEGG pathways passed the minimum pathway-size filter")

    # Legacy hit-only BH: remove zero-hit pathways before multiple testing.
    zero_hit = res_all[res_all["candidate_hits"] == 0].copy()
    zero_hit["BH_treatment"] = "Excluded before BH because candidate_hits == 0"
    zero_hit.to_csv(outdir / "KEGG_zero_hit_excluded_from_BH.csv", index=False)

    res = res_all[res_all["candidate_hits"] > 0].copy()
    if res.empty:
        raise ValueError("No KEGG pathway has >=1 top-candidate hit after filtering")

    res["q_value"] = bh_adjust(res["p_value"])
    res["status"] = res["q_value"].apply(q_status)
    res["n_hypotheses_eligible_before_hit_filter"] = len(res_all)
    res["n_hypotheses_BH"] = len(res)

    memb_rows = []
    for pid in sorted(tested_paths):
        for g in sorted(path2genes[pid]):
            memb_rows.append({
                "pathway_id": pid,
                "Description": path_name.get(pid, pid),
                "gene_id": g,
                "is_top200": g in top_set,
                "is_top200_and_KEGG_eligible": g in top_eligible,
            })
    pd.DataFrame(memb_rows).to_csv(outdir / "KEGG_pathway_membership_long.csv", index=False)

    res = res.sort_values(["q_value", "p_value"])
    res.to_csv(outdir / "Table_S4_KEGG_all_tested.csv", index=False)
    res[res["q_value"] < 0.10].to_csv(
        outdir / "Table_S4_KEGG_reported_q_lt_0.10.csv", index=False
    )

    return {
        "analysis": "KEGG",
        "ranking_universe_n": len(base_universe),
        "top_candidates_requested_n": len(top_set),
        "eligible_background_n": M,
        "eligible_top_candidates_n": N,
        "mapping_success_fraction_top": N / len(top_set),
        "min_term_size": min_path_size,
        "hypotheses_all_n": len(res_all),
        "hypotheses_primary_BH_n": len(res),
        "hypotheses_zero_hit_excluded_n": int((res_all["candidate_hits"] == 0).sum()),
        "hypotheses_hit_terms_n": len(res),
        "excluded_seed_terms_requested_n": 0,
        "significant_q_lt_0.05_n": int((res["q_value"] < 0.05).sum()),
        "descriptive_0.05_le_q_lt_0.10_n": int(((res["q_value"] >= 0.05) & (res["q_value"] < 0.10)).sum()),
        "background_definition": "genes in ensemble_ranking.csv mapped to >=1 P. trichocarpa KEGG pathway",
        "candidate_definition": "top-ranked 200 genes intersected with KEGG eligible background",
        "BH_definition": "legacy hit-only BH: after min-size filtering, remove candidate_hits==0 pathways, then BH across remaining hit pathways",
    }


# -----------------------------------------------------------------------------
# TF family
# -----------------------------------------------------------------------------

def load_tf(path: Path) -> pd.DataFrame:
    df = read_table_flexible(path)
    gene_col = resolve_col(df, ["Gene_ID", "gene_id", "Gene", "gene"])
    family_col = resolve_col(df, ["Family", "family", "TF_family", "TF Family"])
    out = pd.DataFrame({
        "gene_id": df[gene_col].apply(norm_gene_id),
        "Family": df[family_col].astype(str).str.strip(),
    })
    out = out.dropna(subset=["gene_id", "Family"])
    out = out[(out["Family"] != "") & (out["Family"].str.lower() != "nan")]
    return out.drop_duplicates(["gene_id", "Family"]).reset_index(drop=True)


def run_tf(
    tf: pd.DataFrame,
    base_universe: Set[str],
    top_set: Set[str],
    min_family_size: int,
    outdir: Path,
) -> Dict[str, object]:
    # A genome-wide TF list is interpreted as a membership list. Therefore all ranked
    # genes are eligible; genes absent from the list are non-TFs, not "unmappable".
    eligible = set(base_universe)
    top_eligible = set(top_set)
    M, N = len(eligible), len(top_eligible)
    tf = tf[tf["gene_id"].isin(eligible)].copy()
    tf_genes = set(tf["gene_id"])

    write_gene_list(outdir / "eligible_universe_TF.txt", eligible)
    write_gene_list(outdir / "top200_eligible_TF.txt", top_eligible)

    overall_hits = top_eligible & tf_genes
    overall_stat = fisher_stats(len(overall_hits), N, len(tf_genes), M)
    overall_df = pd.DataFrame([{
        "analysis": "Any transcription factor",
        "candidate_hits": len(overall_hits),
        "candidate_eligible_n": N,
        "background_members": len(tf_genes),
        "background_eligible_n": M,
        **overall_stat,
        "q_value": overall_stat["p_value"],  # one prespecified overall hypothesis
        "status": q_status(overall_stat["p_value"]),
        "hit_genes": ";".join(sorted(overall_hits)),
    }])
    overall_df.to_csv(outdir / "TF_overall_enrichment.csv", index=False)

    fam2genes = tf.groupby("Family")["gene_id"].apply(set).to_dict()
    rows: List[Dict[str, object]] = []
    tested_fams: Set[str] = set()
    for fam in sorted(fam2genes):
        genes = fam2genes[fam] & eligible
        K = len(genes)
        if K < min_family_size:
            continue
        tested_fams.add(fam)
        hits = genes & top_eligible
        stat = fisher_stats(len(hits), N, K, M)
        rows.append({
            "Family": fam,
            "candidate_hits": len(hits),
            "candidate_eligible_n": N,
            "background_members": K,
            "background_eligible_n": M,
            **stat,
            "hit_genes": ";".join(sorted(hits)),
        })

    res_all = pd.DataFrame(rows)
    if res_all.empty:
        raise ValueError("No TF families passed the minimum family-size filter")

    # Legacy hit-only BH: remove zero-hit TF families before multiple testing.
    zero_hit = res_all[res_all["candidate_hits"] == 0].copy()
    zero_hit["BH_treatment"] = "Excluded before BH because candidate_hits == 0"
    zero_hit.to_csv(outdir / "TF_family_zero_hit_excluded_from_BH.csv", index=False)

    res = res_all[res_all["candidate_hits"] > 0].copy()
    if res.empty:
        raise ValueError("No TF family has >=1 top-candidate hit after filtering")

    res["q_value"] = bh_adjust(res["p_value"])
    res["status"] = res["q_value"].apply(q_status)
    res["n_hypotheses_eligible_before_hit_filter"] = len(res_all)
    res["n_hypotheses_BH"] = len(res)

    membership = tf[tf["Family"].isin(tested_fams)][["Family", "gene_id"]].drop_duplicates()
    membership["is_top200"] = membership["gene_id"].isin(top_set)
    membership.to_csv(outdir / "TF_family_membership_long.csv", index=False)

    res = res.sort_values(["q_value", "p_value"])
    res.to_csv(outdir / "Table_S5_TF_family_all_tested.csv", index=False)
    res[res["q_value"] < 0.10].to_csv(
        outdir / "Table_S5_TF_family_reported_q_lt_0.10.csv", index=False
    )

    return {
        "analysis": "TF-family",
        "ranking_universe_n": len(base_universe),
        "top_candidates_requested_n": len(top_set),
        "eligible_background_n": M,
        "eligible_top_candidates_n": N,
        "mapping_success_fraction_top": 1.0,
        "min_term_size": min_family_size,
        "hypotheses_all_n": len(res_all),
        "hypotheses_primary_BH_n": len(res),
        "hypotheses_zero_hit_excluded_n": int((res_all["candidate_hits"] == 0).sum()),
        "hypotheses_hit_terms_n": len(res),
        "excluded_seed_terms_requested_n": 0,
        "significant_q_lt_0.05_n": int((res["q_value"] < 0.05).sum()),
        "descriptive_0.05_le_q_lt_0.10_n": int(((res["q_value"] >= 0.05) & (res["q_value"] < 0.10)).sum()),
        "background_definition": "all genes in ensemble_ranking.csv; Ptr_TF_list is treated as a TF-membership census",
        "candidate_definition": "top-ranked 200 genes",
        "BH_definition": "legacy hit-only BH: after min-size filtering, remove candidate_hits==0 TF families, then BH across remaining hit families",
    }


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def make_top200_audit(
    ranking: pd.DataFrame,
    top_n: int,
    go_eligible: Set[str],
    kegg_eligible: Set[str],
    tf_genes: Set[str],
    outdir: Path,
) -> None:
    top = ranking.head(top_n).copy()
    top["GO_BP_eligible"] = top["gene_id"].isin(go_eligible)
    top["KEGG_eligible"] = top["gene_id"].isin(kegg_eligible)
    top["is_TF"] = top["gene_id"].isin(tf_genes)
    top.to_csv(outdir / "top200_candidate_annotation_audit.csv", index=False)


def make_simple_go_plot(outdir: Path) -> None:
    """Optional compact Figure-4 replacement using only q<0.10 reported GO rows."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warning] matplotlib unavailable; skipping GO plot: {e}")
        return

    path = outdir / "Table_S3_GO_BP_reported_q_lt_0.10.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    if d.empty:
        return
    d = d.sort_values(["q_value", "fold_enrichment"]).head(20).copy().iloc[::-1]
    labels = [
        f"{t} ({gid})" + (" *" if q < 0.05 else " †")
        for t, gid, q in zip(d["GO_term"], d["GO_ID"], d["q_value"])
    ]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(d) + 1.8)))
    ax.barh(labels, d["fold_enrichment"])
    ax.set_xlabel("Fold enrichment")
    ax.set_title("GO-BP enrichment among top-200 candidates\n(annotation-eligible ranked-gene background)")
    for i, (_, row) in enumerate(d.iterrows()):
        ax.text(float(row["fold_enrichment"]), i, f"  q={row['q_value']:.3g}", va="center", fontsize=8)
    fig.text(0.01, 0.01, "* q < 0.05; † 0.05 ≤ q < 0.10 (descriptive trend only)", fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(outdir / "Figure4_GO_BP_revised.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reviewer-ready GO/KEGG/TF enrichment from final ensemble ranking")
    p.add_argument("--ranking", required=True, type=Path)
    p.add_argument("--go", required=True, type=Path, help="GO annotation table")
    p.add_argument("--kegg-gene-ko", required=True, type=Path, help="Potri gene -> KO annotation table")
    p.add_argument("--keg", required=True, type=Path, help="pop00001.keg")
    p.add_argument("--tf", required=True, type=Path, help="Genome-wide TF membership/family table")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--top-n", type=int, default=200)
    p.add_argument("--go-min-term-size", type=int, default=5)
    p.add_argument("--kegg-min-path-size", type=int, default=3)
    p.add_argument("--tf-min-family-size", type=int, default=5)
    p.add_argument(
        "--go-seed-terms", nargs="*", default=["GO:0009631", "GO:0050826","GO:0009409", "GO:0071497"],
        help="Complete revised set of GO IDs used directly for seed construction",
    )
    p.add_argument(
        "--go-seed-term-file", type=Path, default=None,
        help="Optional text/CSV-like file; every GO:NNNNNNN token found is added to --go-seed-terms",
    )
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    seed_terms = {str(x).strip() for x in args.go_seed_terms if str(x).strip()}
    if args.go_seed_term_file is not None:
        text = args.go_seed_term_file.read_text(encoding="utf-8", errors="ignore")
        seed_terms.update(re.findall(r"GO:\d+", text))
    bad = sorted(x for x in seed_terms if not re.fullmatch(r"GO:\d+", x))
    if bad:
        raise ValueError(f"Invalid GO seed-term IDs: {bad}")

    ranking, base_universe, top_list, top_set = load_ranking(args.ranking, args.top_n)
    print(f"[ranking] candidate universe={len(base_universe)}, top={len(top_set)}")
    ranking.to_csv(args.outdir / "ensemble_ranking_used.csv", index=False)
    pd.DataFrame({"rank_within_top": np.arange(1, args.top_n + 1), "gene_id": top_list}).to_csv(
        args.outdir / "top200_candidates_used.csv", index=False
    )

    go = load_go(args.go)
    go_summary = run_go(
        go, base_universe, top_set, seed_terms, args.go_min_term_size, args.outdir
    )
    print(f"[GO] eligible background={go_summary['eligible_background_n']}, "
          f"eligible top={go_summary['eligible_top_candidates_n']}, "
          f"primary hypotheses={go_summary['hypotheses_primary_BH_n']}")

    ko2path, path_name = parse_keg(args.keg)
    g2k = load_gene2ko(args.kegg_gene_ko)
    kegg_summary = run_kegg(
        g2k, ko2path, path_name, base_universe, top_set, args.kegg_min_path_size, args.outdir
    )
    print(f"[KEGG] eligible background={kegg_summary['eligible_background_n']}, "
          f"eligible top={kegg_summary['eligible_top_candidates_n']}, "
          f"hypotheses={kegg_summary['hypotheses_primary_BH_n']}")

    tf = load_tf(args.tf)
    tf_genes = set(tf["gene_id"]) & base_universe
    tf_summary = run_tf(
        tf, base_universe, top_set, args.tf_min_family_size, args.outdir
    )
    print(f"[TF] background={tf_summary['eligible_background_n']}, top={tf_summary['eligible_top_candidates_n']}, "
          f"families={tf_summary['hypotheses_primary_BH_n']}")

    # Reconstruct analysis-specific eligible sets for the top-200 mapping audit.
    go_eligible = set(go.loc[go["gene_id"].isin(base_universe), "gene_id"])
    gene2path = {}
    for g in base_universe:
        pset = set()
        for ko in g2k.get(g, set()):
            pset.update(ko2path.get(ko, set()))
        if pset:
            gene2path[g] = pset
    kegg_eligible = set(gene2path)
    make_top200_audit(ranking, args.top_n, go_eligible, kegg_eligible, tf_genes, args.outdir)

    summary = pd.DataFrame([go_summary, kegg_summary, tf_summary])
    summary["inferential_threshold"] = "q < 0.05"
    summary["descriptive_trend_range"] = "0.05 <= q < 0.10"
    summary.to_csv(args.outdir / "enrichment_analysis_summary.csv", index=False)

    manifest = {
        "ranking_file": str(args.ranking),
        "ranking_candidate_universe_n": len(base_universe),
        "top_n": args.top_n,
        "go_annotation": str(args.go),
        "kegg_gene_to_ko": str(args.kegg_gene_ko),
        "kegg_keg": str(args.keg),
        "tf_annotation": str(args.tf),
        "GO_seed_construction_terms_excluded_from_primary_BH": sorted(seed_terms),
        "GO_primary_multiple_testing_treatment": (
            "Seed-construction GO terms are excluded a priori from the primary BH hypothesis family. "
            "Their raw P values and q_value_all_terms are retained for audit; primary q_value is NA."
        ),
        "BH_zero_hit_policy": "Legacy hit-only BH: hypotheses with candidate_hits == 0 are excluded before BH adjustment.",
        "inferential_threshold": "q < 0.05",
        "descriptive_only": "0.05 <= q < 0.10",
        "effect_sizes": ["fold_enrichment", "odds_ratio_fisher", "odds_ratio_haldane", "Haldane approximate 95% CI"],
    }
    (args.outdir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_plot:
        make_simple_go_plot(args.outdir)

    print(f"[done] reviewer-ready enrichment outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()

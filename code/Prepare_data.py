"""Prepare label-independent inputs for PoplarCold-PUGAT."""

import argparse
from pathlib import Path


def parse_args():
    repo_dir = Path(__file__).resolve().parent
    default_data_dir = repo_dir / "data"

    parser = argparse.ArgumentParser(
        description="Prepare expression features and co-expression/PPI graphs for PoplarCold-PUGAT."
    )
    parser.add_argument("--data_dir", type=Path, default=default_data_dir,
                        help="Directory containing the raw Expression Atlas files, PPI table, and audited seed table.")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Output directory. If omitted, a threshold-descriptive directory is created under --data_dir.")
    parser.add_argument("--raw_counts_tsv", type=Path, default=None)
    parser.add_argument("--de_table_tsv", type=Path, default=None)
    parser.add_argument("--positives_xlsx", type=Path, default=None,
                        help="Audited positive-seed spreadsheet. Defaults to Supplementary_Data_1_revised_final.xlsx in --data_dir.")
    parser.add_argument("--ppi_csv", type=Path, default=None)
    parser.add_argument("--ppi_score_threshold", type=int, default=800)
    parser.add_argument("--coexp_corr_threshold", type=float, default=0.75)
    parser.add_argument("--coexp_top_mr", type=int, default=15)
    parser.add_argument("--coexp_chunk", type=int, default=512)
    return parser.parse_args()


ARGS = parse_args()
DATA_DIR = ARGS.data_dir.expanduser().resolve()
PPI_SCORE_THRESHOLD = int(ARGS.ppi_score_threshold)
COEXP_CORR_THRESHOLD = float(ARGS.coexp_corr_threshold)
COEXP_TOP_MR = int(ARGS.coexp_top_mr)
COEXP_CHUNK = int(ARGS.coexp_chunk)

if ARGS.output_dir is None:
    OUTPUT_DIR = DATA_DIR / (
        f"preprocessed_label_independent_ppi{PPI_SCORE_THRESHOLD}_corr{COEXP_CORR_THRESHOLD:g}"
    )
else:
    OUTPUT_DIR = ARGS.output_dir.expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_COUNTS_TSV = (ARGS.raw_counts_tsv or (DATA_DIR / "E-MTAB-5540-raw-counts.tsv")).expanduser().resolve()
DE_TABLE_TSV = (ARGS.de_table_tsv or (DATA_DIR / "E-MTAB-5540-analytics.tsv")).expanduser().resolve()
POSITIVES_XLSX = (ARGS.positives_xlsx or (DATA_DIR / "Supplementary_Data_1_revised_final.xlsx")).expanduser().resolve()
PPI_CSV = (ARGS.ppi_csv or (DATA_DIR / "PPI.csv")).expanduser().resolve()

MIN_CPM = 1.0
MIN_SAMPLES = 5
DEG_LOG2FC = 1.0
DEG_P = 0.05
BUILD_COEXP_GRAPH = True
NODE_INCLUSION_RULE = "any_graph_edge"

print("OUTPUT_DIR =", OUTPUT_DIR)

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

stats_lines = []
def log(msg=""):
    print(msg)
    stats_lines.append(str(msg))

def norm_gene_id(x):
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

def parse_condition_name(name):
    name = str(name).strip()
    m = re.match(r"'([^']+)' vs 'control' in '([^']+)'", name)
    if not m:
        return None
    treat, tissue = m.group(1), m.group(2)
    t = treat.lower()
    if "cold" in t:
        stress = "cold"
    elif "drought" in t:
        stress = "drought"
    elif "heat" in t:
        stress = "heat"
    elif "salt" in t:
        stress = "salt"
    else:
        stress = "unknown"
    duration = "prolonged" if "prolonged" in t else "short"
    return stress, duration, tissue.lower().strip()

def read_de_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DE table not found: {path}")
    header_line = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("Gene ID\t"):
                header_line = i
                break
    df = pd.read_csv(path, sep="\t", skiprows=header_line, low_memory=False)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    if "Gene ID" not in df.columns:
        raise ValueError("DE table missing Gene ID column.")
    df["Potri_ID"] = df["Gene ID"].apply(norm_gene_id)
    df = df[df["Potri_ID"].notna()].drop_duplicates("Potri_ID").reset_index(drop=True)
    return df

def split_de_columns(df):
    cond_to_fc, cond_to_pv = {}, {}
    for col in df.columns:
        c_low = col.lower().strip()
        is_fc = c_low.endswith(".log2foldchange") or c_low.endswith(".foldchange")
        is_pv = c_low.endswith(".p-value") or c_low.endswith(".pvalue") or c_low.endswith(".p.value")
        if not (is_fc or is_pv):
            continue
        base = col
        base = re.sub(r"\.log2foldchange$", "", base, flags=re.I)
        base = re.sub(r"\.foldchange$", "", base, flags=re.I)
        base = re.sub(r"\.p-value$", "", base, flags=re.I)
        base = re.sub(r"\.pvalue$", "", base, flags=re.I)
        base = re.sub(r"\.p\.value$", "", base, flags=re.I)
        parsed = parse_condition_name(base.strip())
        if parsed is None:
            continue
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
        if is_fc:
            cond_to_fc[parsed] = values
        else:
            cond_to_pv[parsed] = values
    return cond_to_fc, cond_to_pv

def align_values_to_genes(values, qdf, gene_ids, fill):
    idx = pd.Series(np.arange(len(qdf)), index=qdf["Potri_ID"]).to_dict()
    out = np.full(len(gene_ids), fill, dtype=np.float32)
    for i, gid in enumerate(gene_ids):
        j = idx.get(gid)
        if j is not None:
            v = values[j]
            if not np.isnan(v):
                out[i] = v
    return out

def save_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, lines)))

log("=" * 80)
log("Step 1: Read raw counts and calculate logCPM")
log("=" * 80)

raw = pd.read_csv(RAW_COUNTS_TSV, sep="\t", low_memory=False)
raw.columns = [str(c).strip().lstrip("\ufeff") for c in raw.columns]
if "Gene ID" not in raw.columns:
    raise ValueError("Raw counts file missing Gene ID column.")

raw["Potri_ID"] = raw["Gene ID"].apply(norm_gene_id)
raw = raw[raw["Potri_ID"].notna()].drop_duplicates("Potri_ID").reset_index(drop=True)

sample_cols = [c for c in raw.columns if str(c).startswith("ERR")]
if len(sample_cols) == 0:
    raise ValueError("No sample columns starting with ERR found.")

gene_ids_all = raw["Potri_ID"].astype(str).tolist()
count_matrix = raw[sample_cols].to_numpy(dtype=np.float32)
lib_sizes = count_matrix.sum(axis=0)

cpm = count_matrix / lib_sizes[None, :] * 1e6
log_cpm = np.log2(cpm + 1.0).astype(np.float32)
expressed_mask = (cpm >= MIN_CPM).sum(axis=1) >= MIN_SAMPLES

log(f"raw shape: {raw.shape}")
log(f"Potri genes: {len(gene_ids_all)}")
log(f"samples: {len(sample_cols)}")
log(f"library size: min={lib_sizes.min():,.0f}, max={lib_sizes.max():,.0f}, mean={lib_sizes.mean():,.0f}")
log(f"expressed genes: {int(expressed_mask.sum())} / {len(gene_ids_all)}")

log()
log("=" * 80)
log("Step 2: Read ExpressionAtlas DE/analytics table")
log("=" * 80)

qdf = read_de_table(DE_TABLE_TSV)
cond_to_fc_raw, cond_to_pv_raw = split_de_columns(qdf)

log(f"DE table Potri genes: {len(qdf)}")
log(f"parsed foldchange comparisons: {len(cond_to_fc_raw)}")
log(f"parsed p-value comparisons: {len(cond_to_pv_raw)}")

cond_to_fc = {k: align_values_to_genes(v, qdf, gene_ids_all, fill=0.0) for k, v in cond_to_fc_raw.items()}
cond_to_pv = {k: align_values_to_genes(v, qdf, gene_ids_all, fill=1.0) for k, v in cond_to_pv_raw.items()}

cold_keys = sorted([k for k in cond_to_fc if k[0] == "cold"])
other_keys = sorted([k for k in cond_to_fc if k[0] != "cold"])

log(f"cold comparisons: {len(cold_keys)}")
for k in cold_keys:
    log(f"  {k}")
log(f"other stress comparisons: {len(other_keys)}")

if len(cold_keys) != 6:
    raise ValueError(f"Expected 6 cold conditions, found {len(cold_keys)}.")
for k in cold_keys:
    if k not in cond_to_pv:
        raise ValueError(f"Missing p-value for condition {k}.")

log()
log("=" * 80)
log("Step 3: Construct 43 node features")
log("=" * 80)

cold_fc_mat = np.stack([cond_to_fc[k] for k in cold_keys], axis=1).astype(np.float32)
cold_pv_mat = np.stack([cond_to_pv[k] for k in cold_keys], axis=1).astype(np.float32)
cold_negp_mat = -np.log10(np.clip(cold_pv_mat, 1e-30, 1.0)).astype(np.float32)
other_fc_mat = np.stack([cond_to_fc[k] for k in other_keys], axis=1).astype(np.float32)

max_abs_cold_fc = np.max(np.abs(cold_fc_mat), axis=1)
mean_abs_cold_fc = np.mean(np.abs(cold_fc_mat), axis=1)
max_neglog_p_cold = np.max(cold_negp_mat, axis=1)
cold_dir_consistency = np.abs(cold_fc_mat.sum(axis=1)) / (np.abs(cold_fc_mat).sum(axis=1) + 1e-6)

mean_abs_other_fc = np.mean(np.abs(other_fc_mat), axis=1)
cold_specificity = mean_abs_cold_fc - mean_abs_other_fc
cold_specificity_z = (mean_abs_cold_fc - mean_abs_other_fc) / (np.std(other_fc_mat, axis=1) + 0.1)

short_idx = [i for i, k in enumerate(cold_keys) if k[1] == "short"]
long_idx = [i for i, k in enumerate(cold_keys) if k[1] == "prolonged"]
short_fc = cold_fc_mat[:, short_idx]
long_fc = cold_fc_mat[:, long_idx]
short_term_tissue_consis = np.abs(short_fc.sum(axis=1)) / (np.abs(short_fc).sum(axis=1) + 1e-6)
prolonged_tissue_consis = np.abs(long_fc.sum(axis=1)) / (np.abs(long_fc).sum(axis=1) + 1e-6)

dur_diffs = []
for tissue in ["root", "stem xylem", "vascular leaf"]:
    s_i = next((i for i, k in enumerate(cold_keys) if k[1] == "short" and k[2] == tissue), None)
    p_i = next((i for i, k in enumerate(cold_keys) if k[1] == "prolonged" and k[2] == tissue), None)
    if s_i is not None and p_i is not None:
        dur_diffs.append(cold_fc_mat[:, p_i] - cold_fc_mat[:, s_i])
mean_duration_persistence = np.stack(dur_diffs, axis=1).mean(axis=1) if dur_diffs else np.zeros(len(gene_ids_all), dtype=np.float32)

cold_deg_bool = ((np.abs(cold_fc_mat) >= DEG_LOG2FC) & (cold_pv_mat <= DEG_P))
n_cold_conds_DEG = cold_deg_bool.sum(axis=1).astype(np.float32)
is_strong_cold_DEG = cold_deg_bool.any(axis=1).astype(np.float32)
baseline_log_expr = log_cpm.mean(axis=1).astype(np.float32)

cold_feature_names = (
    [f"cold_fc_{k[1]}_{k[2]}".replace(" ", "_") for k in cold_keys] +
    [f"cold_negp_{k[1]}_{k[2]}".replace(" ", "_") for k in cold_keys]
)
other_feature_names = [f"other_fc_{k[0]}_{k[1]}_{k[2]}".replace(" ", "_") for k in other_keys]
scalar_names_without_graph = [
    "max_abs_cold_fc", "mean_abs_cold_fc", "max_neglog_p_cold",
    "cold_dir_consistency", "cold_specificity", "cold_specificity_z",
    "short_term_tissue_consis", "prolonged_tissue_consis",
    "mean_duration_persistence", "n_cold_conds_DEG",
    "is_strong_cold_DEG", "baseline_log_expr"
]

log(f"cold DEG baseline genes: {int(is_strong_cold_DEG.sum())} / {len(gene_ids_all)}")
log(f"features so far: {len(cold_feature_names)} cold + {len(other_feature_names)} other + {len(scalar_names_without_graph)} scalar_without_graph")

log()
log("=" * 80)
log("Step 4: Read positive genes")
log("=" * 80)

pos_sheets = pd.read_excel(POSITIVES_XLSX, sheet_name=None)
if "Main_Positives_Pt_Genes" in pos_sheets:
    pos_df = pos_sheets["Main_Positives_Pt_Genes"]
else:
    first_sheet = list(pos_sheets.keys())[0]
    log(f"Main_Positives_Pt_Genes not found, using first sheet: {first_sheet}")
    pos_df = pos_sheets[first_sheet]

if "Pt_Gene_ID" not in pos_df.columns:
    raise ValueError("Positive table must contain Pt_Gene_ID column.")

pos_df["Potri_ID"] = pos_df["Pt_Gene_ID"].apply(norm_gene_id)
positive_set = set(pos_df["Potri_ID"].dropna().astype(str))

# The audited Supplementary Data 1 contains only positive seed genes.
# Evidence_Tier records provenance strength/category (A-D) and must not be
# interpreted as a positive/negative class label.  Older releases used a
# Source_Tier field and removed rows containing the word "NEGATIVE"; that
# legacy rule is intentionally removed because a negative regulator of cold
# response can still be a valid cold-associated positive seed.
if "Evidence_Tier" in pos_df.columns:
    evidence_tiers = (
        pos_df["Evidence_Tier"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )
    unexpected_tiers = sorted(set(evidence_tiers) - {"A", "B", "C", "D"})
    if unexpected_tiers:
        log(f"warning: unexpected Evidence_Tier values: {unexpected_tiers}")

y_all = np.array([1 if gid in positive_set else 0 for gid in gene_ids_all], dtype=np.int64)

log(f"known positives after filtering: {len(positive_set)}")
log(f"matched positives: {int(y_all.sum())}")

pos_signal = pd.DataFrame({
    "gene_id": gene_ids_all,
    "is_positive": y_all,
    "max_abs_cold_fc": max_abs_cold_fc,
    "max_neglog_p_cold": max_neglog_p_cold,
    "cold_specificity": cold_specificity,
    "n_cold_conds_DEG": n_cold_conds_DEG,
    "is_strong_cold_DEG": is_strong_cold_DEG,
    "baseline_log_expr": baseline_log_expr,
})
pos_signal[pos_signal["is_positive"] == 1].to_csv(OUTPUT_DIR / "positives_signal_report.csv", index=False)

log()
log("=" * 80)
log("Step 5: Build co-expression graph")
log("=" * 80)

n_genes_all = len(gene_ids_all)
coexp_edges = np.zeros((2, 0), dtype=np.int64)
coexp_w = np.zeros((0,), dtype=np.float32)
coexp_degree = np.zeros(n_genes_all, dtype=np.float32)

if BUILD_COEXP_GRAPH:
    idx_to_orig = np.where(expressed_mask)[0]
    expr = log_cpm[expressed_mask]
    n_expr = expr.shape[0]
    log(f"genes used for coexpression: {n_expr}")

    ranked = np.apply_along_axis(rankdata, 1, expr).astype(np.float32)
    ranked = (ranked - ranked.mean(axis=1, keepdims=True)) / (ranked.std(axis=1, keepdims=True) + 1e-8)

    edges_src, edges_tgt, edges_weight = [], [], []
    for cs in range(0, n_expr, COEXP_CHUNK):
        ce = min(cs + COEXP_CHUNK, n_expr)
        corr = (ranked[cs:ce] @ ranked.T) / ranked.shape[1]
        for local_i in range(corr.shape[0]):
            i = cs + local_i
            abs_corr = np.abs(corr[local_i])
            abs_corr[i] = 0.0
            top_k = min(COEXP_TOP_MR, n_expr - 1)
            top_idx = np.argpartition(-abs_corr, top_k)[:top_k]
            top_idx = top_idx[abs_corr[top_idx] >= COEXP_CORR_THRESHOLD]
            for j in top_idx:
                edges_src.append(i)
                edges_tgt.append(int(j))
                edges_weight.append(float(abs_corr[j]))
        if ce % (COEXP_CHUNK * 5) == 0 or ce == n_expr:
            log(f"  processed {ce}/{n_expr}, directed candidate edges={len(edges_src)}")

    edge_lookup = set(zip(edges_src, edges_tgt))
    mutual = [(s, t, w) for s, t, w in zip(edges_src, edges_tgt, edges_weight)
              if s < t and (t, s) in edge_lookup]
    log(f"mutual coexpression undirected edges: {len(mutual)}")

    if len(mutual) > 0:
        src = np.array([idx_to_orig[s] for s, t, w in mutual], dtype=np.int64)
        tgt = np.array([idx_to_orig[t] for s, t, w in mutual], dtype=np.int64)
        w = np.array([w for s, t, w in mutual], dtype=np.float32)
        coexp_edges = np.vstack([np.concatenate([src, tgt]), np.concatenate([tgt, src])]).astype(np.int64)
        coexp_w = np.concatenate([w, w]).astype(np.float32)
        np.add.at(coexp_degree, coexp_edges[0], 1.0)

log(f"coexpression directed edges: {coexp_edges.shape[1]}")

log()
log("=" * 80)
log("Step 6: Build PPI graph")
log("=" * 80)

ppi_df = pd.read_csv(PPI_CSV, sep=None, engine="python")
ppi_df.columns = [str(c).strip().lstrip("\ufeff") for c in ppi_df.columns]
required = {"gene1", "gene2", "combined_score"}
if not required.issubset(set(ppi_df.columns)):
    raise ValueError(f"PPI file requires columns: {required}, current columns: {list(ppi_df.columns)}")

ppi_df["g1"] = ppi_df["gene1"].apply(norm_gene_id)
ppi_df["g2"] = ppi_df["gene2"].apply(norm_gene_id)
ppi_df["combined_score"] = pd.to_numeric(ppi_df["combined_score"], errors="coerce")
ppi_df = ppi_df.dropna(subset=["g1", "g2", "combined_score"])

# Filter rule: keep only edges with combined_score >= PPI_SCORE_THRESHOLD
ppi_use_df = ppi_df[ppi_df["combined_score"] >= PPI_SCORE_THRESHOLD].copy()

log(f"PPI raw edges: {len(ppi_df)}")
log(f"PPI edges kept (score>={PPI_SCORE_THRESHOLD}): {len(ppi_use_df)}")
log(f"PPI edges used as graph before mapping/dedup: {len(ppi_use_df)}")

# Map gene IDs to indices and build directed graph
gene_to_idx = {g: i for i, g in enumerate(gene_ids_all)}
def _ppi_to_directed(df_in):
    if len(df_in) == 0:
        return (np.empty((2, 0), dtype=np.int64),
                np.empty((0,), dtype=np.float32))
    d = df_in.copy()
    d["i1"] = d["g1"].map(gene_to_idx)
    d["i2"] = d["g2"].map(gene_to_idx)
    d = d.dropna(subset=["i1", "i2"])
    d["i1"] = d["i1"].astype(int)
    d["i2"] = d["i2"].astype(int)
    d = d[d["i1"] != d["i2"]].copy()
    d["u"] = np.minimum(d["i1"], d["i2"])
    d["v"] = np.maximum(d["i1"], d["i2"])
    d = d.groupby(["u", "v"], as_index=False)["combined_score"].max()
    src = d["u"].to_numpy(dtype=np.int64)
    tgt = d["v"].to_numpy(dtype=np.int64)
    w = d["combined_score"].to_numpy(dtype=np.float32) / 1000.0
    edges = np.vstack([np.concatenate([src, tgt]),
                       np.concatenate([tgt, src])]).astype(np.int64)
    weight = np.concatenate([w, w]).astype(np.float32)
    return edges, weight

ppi_edges, ppi_w = _ppi_to_directed(ppi_use_df)

ppi_degree = np.zeros(n_genes_all, dtype=np.float32)
if ppi_edges.shape[1] > 0:
    np.add.at(ppi_degree, ppi_edges[0], 1.0)

# Sanity check: PPI degree distribution
pos_mask_global = (y_all == 1)
pos_deg = ppi_degree[pos_mask_global]
unl_deg = ppi_degree[~pos_mask_global]
log(f"[Sanity] PPI degree -- positives: mean={pos_deg.mean():.2f}, median={np.median(pos_deg):.1f}, max={pos_deg.max():.0f}")
log(f"[Sanity] PPI degree -- unlabeled: mean={unl_deg.mean():.2f}, median={np.median(unl_deg):.1f}, max={unl_deg.max():.0f}")

log(f"PPI directed edges: {ppi_edges.shape[1]}")

log()
log("=" * 80)
log("Step 7: Concatenate features, filter nodes, save results")
log("=" * 80)

log_coexp_degree = np.log2(coexp_degree + 1.0).astype(np.float32)
log_ppi_degree = np.log2(ppi_degree + 1.0).astype(np.float32)

cold_features = np.concatenate([cold_fc_mat, cold_negp_mat], axis=1)
other_features = other_fc_mat
scalar_features = np.column_stack([
    max_abs_cold_fc, mean_abs_cold_fc, max_neglog_p_cold,
    cold_dir_consistency, cold_specificity, cold_specificity_z,
    short_term_tissue_consis, prolonged_tissue_consis,
    mean_duration_persistence, n_cold_conds_DEG,
    is_strong_cold_DEG, baseline_log_expr,
    log_coexp_degree, log_ppi_degree,
]).astype(np.float32)

scalar_names = scalar_names_without_graph + ["log_coexp_degree", "log_ppi_degree"]
X_raw_all = np.concatenate([cold_features, other_features, scalar_features], axis=1).astype(np.float32)
feature_names = cold_feature_names + other_feature_names + scalar_names

assert X_raw_all.shape[1] == 43, f"Current feature count: {X_raw_all.shape[1]}, expected 43."
log(f"X_raw_all shape: {X_raw_all.shape}")

# -------------------------------------------------------------------------
# Label-independent node inclusion rule (reviewer Major Comment 3)
# -------------------------------------------------------------------------
# IMPORTANT: the inclusion decision must not depend on y/positive status.
# A node is retained iff it has at least one edge in either input network.
# Therefore, known positives that are edgeless in BOTH networks are removed
# exactly like edgeless unlabeled nodes.
has_coexp = coexp_degree > 0
has_ppi = ppi_degree > 0
has_any_graph_edge = has_coexp | has_ppi

is_pos = y_all == 1
edgeless = ~has_any_graph_edge
affected_pos = is_pos & edgeless
keep = has_any_graph_edge.copy()

if NODE_INCLUSION_RULE != "any_graph_edge":
    raise ValueError(f"Unsupported NODE_INCLUSION_RULE: {NODE_INCLUSION_RULE}")

log(f"node inclusion rule: {NODE_INCLUSION_RULE} (label-independent)")
log(f"nodes with any graph edge: {int(has_any_graph_edge.sum())} / {n_genes_all}")
log(f"edgeless nodes removed (all labels): {int(edgeless.sum())}")
log(f"KNOWN POSITIVES affected by label-independent filter: {int(affected_pos.sum())}")
log(f"known positives before filter: {int(is_pos.sum())}")
log(f"known positives after filter: {int((is_pos & keep).sum())}")
log(f"nodes kept after label-independent filtering: {int(keep.sum())} / {n_genes_all}")

# Save a complete audit table so the exact affected positives can be reported.
node_filter_audit = pd.DataFrame({
    "gene_id": gene_ids_all,
    "is_known_positive": is_pos.astype(int),
    "coexp_degree": coexp_degree.astype(int),
    "ppi_degree": ppi_degree.astype(int),
    "has_coexp_edge": has_coexp.astype(int),
    "has_ppi_edge": has_ppi.astype(int),
    "has_any_graph_edge": has_any_graph_edge.astype(int),
    "removed_as_edgeless": edgeless.astype(int),
    "kept": keep.astype(int),
})

# Remap indices
old_to_new = -np.ones(n_genes_all, dtype=np.int64)
old_indices = np.where(keep)[0]
old_to_new[old_indices] = np.arange(len(old_indices))

def remap_edges(edge_index, edge_weight):
    if edge_index.shape[1] == 0:
        return edge_index, edge_weight
    s = old_to_new[edge_index[0]]
    t = old_to_new[edge_index[1]]
    valid = (s >= 0) & (t >= 0)
    return np.vstack([s[valid], t[valid]]).astype(np.int64), edge_weight[valid].astype(np.float32)

# Filter data
X_raw = X_raw_all[keep]
y = y_all[keep]
gene_ids = [gene_ids_all[i] for i in old_indices]
cond_input = cold_fc_mat[keep].astype(np.float32)
other_input = other_fc_mat[keep].astype(np.float32)

# Remap graph edges
coexp_edges_f, coexp_w_f = remap_edges(coexp_edges, coexp_w)
ppi_edges_f, ppi_w_f = remap_edges(ppi_edges, ppi_w)

# No label-dependent self-loops are added. Nodes that were edgeless in both
# graphs have already been removed by the same rule regardless of label.
log(f"coexp directed edges after label-independent filter: {coexp_edges_f.shape[1]}")
log(f"PPI directed edges after label-independent filter: {ppi_edges_f.shape[1]}")

# Standardize features
binary_names = {"is_strong_cold_DEG"}
binary_idx = [i for i, n in enumerate(feature_names) if n in binary_names]
cont_idx = [i for i in range(X_raw.shape[1]) if i not in binary_idx]

scaler = StandardScaler()
X = X_raw.copy()
X[:, cont_idx] = scaler.fit_transform(X_raw[:, cont_idx]).astype(np.float32)

# Create feature DataFrame
feature_df = pd.DataFrame(X_raw, columns=feature_names)
feature_df.insert(0, "gene_id", gene_ids)
feature_df.insert(1, "y", y)

# DEG score calculation
deg_score_all = max_abs_cold_fc * (1.0 + max_neglog_p_cold) + n_cold_conds_DEG
deg_df_all = pd.DataFrame({
    "gene_id": gene_ids_all,
    "is_known_positive": y_all,
    "is_cold_DEG": is_strong_cold_DEG.astype(int),
    "n_cold_conds_DEG": n_cold_conds_DEG,
    "max_abs_cold_fc": max_abs_cold_fc,
    "max_neglog_p_cold": max_neglog_p_cold,
    "cold_specificity": cold_specificity,
    "deg_score": deg_score_all,
}).sort_values(["is_cold_DEG", "deg_score"], ascending=[False, False])
deg_df_filt = deg_df_all[deg_df_all["gene_id"].isin(set(gene_ids))].copy()

# Feature manifest
manifest = pd.DataFrame([
    {"feature_index": i, "feature_name": n,
     "group": "cold" if n.startswith("cold_") else ("other_stress" if n.startswith("other_") else "scalar")}
    for i, n in enumerate(feature_names)
])

# ============ Save all outputs ============
np.save(OUTPUT_DIR / "X.npy", X.astype(np.float32))
np.save(OUTPUT_DIR / "X_raw.npy", X_raw.astype(np.float32))
np.save(OUTPUT_DIR / "cond_input.npy", cond_input.astype(np.float32))
np.save(OUTPUT_DIR / "other_input.npy", other_input.astype(np.float32))
np.save(OUTPUT_DIR / "y.npy", y.astype(np.int64))

np.save(OUTPUT_DIR / "edges_coexp.npy", coexp_edges_f.astype(np.int64))
np.save(OUTPUT_DIR / "edges_coexp_weight.npy", coexp_w_f.astype(np.float32))

np.save(OUTPUT_DIR / "edges_ppi.npy", ppi_edges_f.astype(np.int64))
np.save(OUTPUT_DIR / "edges_ppi_weight.npy", ppi_w_f.astype(np.float32))

np.save(OUTPUT_DIR / "scaler_mean.npy", scaler.mean_.astype(np.float32))
np.save(OUTPUT_DIR / "scaler_scale.npy", scaler.scale_.astype(np.float32))

save_lines(OUTPUT_DIR / "gene_ids.txt", gene_ids)
save_lines(OUTPUT_DIR / "feature_names.txt", feature_names)
save_lines(OUTPUT_DIR / "binary_idx.txt", binary_idx)

# Save condition metadata
condition_meta = {
    "cold_keys": [list(k) for k in cold_keys],
    "other_keys": [list(k) for k in other_keys],
    "cond_input_dim": int(cond_input.shape[1]),
    "other_input_dim": int(other_input.shape[1]),
}
with open(OUTPUT_DIR / "condition_meta.json", "w", encoding="utf-8") as f:
    json.dump(condition_meta, f, ensure_ascii=False, indent=2)

# Save reports
feature_df.to_csv(OUTPUT_DIR / "features_raw_table.csv", index=False)
manifest.to_csv(OUTPUT_DIR / "feature_manifest.csv", index=False)
node_filter_audit.to_csv(OUTPUT_DIR / "node_filter_audit.csv", index=False)
node_filter_audit[node_filter_audit["is_known_positive"] == 1].to_csv(OUTPUT_DIR / "positive_node_filter_audit.csv", index=False)
deg_df_all.to_csv(OUTPUT_DIR / "cold_DEG_baseline_all_genes.csv", index=False)
deg_df_filt.to_csv(OUTPUT_DIR / "cold_DEG_baseline_filtered_genes.csv", index=False)

# Save summary
summary = {
    "preprocessing_profile": "label_independent_graph_filter",
    "node_inclusion_rule": NODE_INCLUSION_RULE,
    "node_inclusion_uses_labels": False,
    "n_genes_before_filter": int(n_genes_all),
    "n_genes_after_filter": int(len(gene_ids)),
    "n_features": int(X.shape[1]),
    "n_known_positives_before_filter": int(y_all.sum()),
    "n_known_positives_after_filter": int(y.sum()),
    "n_edgeless_nodes_removed": int(edgeless.sum()),
    "n_edgeless_known_positives_removed": int(affected_pos.sum()),
    "n_edgeless_unlabeled_removed": int((edgeless & ~is_pos).sum()),
    "n_nodes_with_any_graph_edge_before_filter": int(has_any_graph_edge.sum()),
    "affected_positive_gene_ids": [gene_ids_all[i] for i in np.where(affected_pos)[0]],
    "coexpression_construction": {
        "input": "log2(CPM+1) among expressed genes",
        "transform": "rank within each gene across samples",
        "similarity": "absolute Pearson correlation of ranks (Spearman-type)",
        "threshold": float(COEXP_CORR_THRESHOLD),
        "neighbor_rule": f"mutual_top_{COEXP_TOP_MR}",
    },
    "n_cold_deg_before_filter": int(is_strong_cold_DEG.sum()),
    "n_coexp_directed_edges_after_filter": int(coexp_edges_f.shape[1]),
    "n_ppi_directed_edges_after_filter": int(ppi_edges_f.shape[1]),
    "ppi_score_threshold": int(PPI_SCORE_THRESHOLD),
    "cold_keys": [list(k) for k in cold_keys],
    "other_keys": [list(k) for k in other_keys],
}
with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

save_lines(OUTPUT_DIR / "stats_report.txt", stats_lines)

log()
log("Processing completed. Main outputs:")
log(f"  X.npy: {X.shape}")
log(f"  cond_input.npy: {cond_input.shape}")
log(f"  other_input.npy: {other_input.shape}")
log(f"  y.npy: {y.shape}, positives={int(y.sum())}")
log(f"  edges_ppi.npy: {ppi_edges_f.shape[1]}")
log(f"  edges_coexp.npy: {coexp_edges_f.shape[1]}")
#!/usr/bin/env python3
"""
Map STRING protein identifiers to Populus POPTR gene identifiers using a UniProt
ID Mapping export.

Expected UniProt mapping columns
--------------------------------
From
Gene Names

The first identifier matching the pattern POPTR_<3 digits>G<digits> is retained
from the Gene Names field. This reproduces the simple first-match parsing rule
used for the historical preprocessing reconstruction.

The script intentionally does NOT apply the STRING score >= 800 filter and does
NOT deduplicate gene pairs. Those operations are performed downstream by
Prepare_data.py, which:
  1. keeps combined_score >= the configured threshold,
  2. removes self-loops after mapping to the model gene universe,
  3. collapses duplicate undirected gene pairs using the maximum score,
  4. stores each retained undirected pair in both directions.

Examples
--------
python map_string_to_poptr.py \
    --string 3694.protein.links.v12.5.txt.gz \
    --mapping idmapping.xlsx \
    --output PPI.csv

To save disk space:
python map_string_to_poptr.py \
    --string 3694.protein.links.v12.5.txt.gz \
    --mapping idmapping.xlsx \
    --output PPI.csv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
from pathlib import Path

import pandas as pd


POPTR_RE = re.compile(r"POPTR_\d{3}G\d+")


def open_input_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def open_output_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return open(path, "wt", encoding="utf-8", newline="")


def read_mapping(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str)
    elif suffix in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t", dtype=str)
    elif suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
    else:
        raise ValueError(
            f"Unsupported mapping format: {path.suffix}. "
            "Use .xlsx, .xls, .csv, .tsv, or .txt."
        )

    df.columns = [str(c).strip() for c in df.columns]
    required = {"From", "Gene Names"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Mapping file must contain columns {sorted(required)}; "
            f"found {list(df.columns)}"
        )
    return df


def first_poptr(gene_names):
    if pd.isna(gene_names):
        return None
    matches = POPTR_RE.findall(str(gene_names))
    return matches[0] if matches else None


def count_poptr_matches(gene_names) -> int:
    if pd.isna(gene_names):
        return 0
    return len(set(POPTR_RE.findall(str(gene_names))))


def strip_taxon_prefix(value: str, prefix: str) -> str:
    value = value.strip()
    if prefix and value.startswith(prefix):
        return value[len(prefix):]
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Map STRING protein.links protein IDs to POPTR gene IDs."
    )
    parser.add_argument("--string", required=True, type=Path,
                        help="STRING protein.links file (.txt or .txt.gz).")
    parser.add_argument("--mapping", required=True, type=Path,
                        help="UniProt ID Mapping export containing 'From' and 'Gene Names'.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output PPI.csv or PPI.csv.gz.")
    parser.add_argument("--taxon-prefix", default="3694.",
                        help="STRING taxon prefix to remove (default: 3694.).")
    args = parser.parse_args()

    mapping = read_mapping(args.mapping).copy()
    mapping["From"] = mapping["From"].fillna("").astype(str).str.strip()
    mapping["Gene Names"] = mapping["Gene Names"].fillna("").astype(str)

    mapping["n_poptr"] = mapping["Gene Names"].apply(count_poptr_matches)
    mapping["gene"] = mapping["Gene Names"].apply(first_poptr)

    n_total = len(mapping)
    n_no_poptr = int((mapping["n_poptr"] == 0).sum())
    n_one = int((mapping["n_poptr"] == 1).sum())
    n_multi = int((mapping["n_poptr"] > 1).sum())

    usable = mapping[(mapping["From"] != "") & mapping["gene"].notna()].copy()

    # If the same submitted accession occurs more than once, all rows must imply
    # the same retained first POPTR ID. Conflicts are stopped rather than guessed.
    conflicts = (
        usable.groupby("From")["gene"]
        .nunique()
        .loc[lambda s: s > 1]
    )
    if len(conflicts):
        examples = ", ".join(conflicts.index[:10])
        raise ValueError(
            f"{len(conflicts)} 'From' accessions map to conflicting first POPTR IDs. "
            f"Examples: {examples}"
        )

    protein_to_gene = (
        usable.drop_duplicates(subset=["From"], keep="first")
        .set_index("From")["gene"]
        .to_dict()
    )

    print("UniProt mapping summary")
    print(f"  rows in mapping file:                  {n_total:,}")
    print(f"  exactly one POPTR match:              {n_one:,}")
    print(f"  multiple POPTR matches (first kept):  {n_multi:,}")
    print(f"  no standard POPTR match:              {n_no_poptr:,}")
    print(f"  usable submitted accessions:          {len(protein_to_gene):,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_string = 0
    n_mapped = 0
    n_missing_p1 = 0
    n_missing_p2 = 0

    with open_input_text(args.string) as inp, open_output_text(args.output) as out:
        header = inp.readline().strip().split()
        required = {"protein1", "protein2", "combined_score"}
        if not required.issubset(header):
            raise ValueError(
                f"Expected STRING columns {sorted(required)}, found: {header}"
            )

        i1 = header.index("protein1")
        i2 = header.index("protein2")
        iscore = header.index("combined_score")

        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(["gene1", "gene2", "combined_score"])

        for line_no, line in enumerate(inp, start=2):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) <= max(i1, i2, iscore):
                raise ValueError(f"Malformed line {line_no}: {line.rstrip()}")

            n_string += 1

            p1 = strip_taxon_prefix(fields[i1], args.taxon_prefix)
            p2 = strip_taxon_prefix(fields[i2], args.taxon_prefix)
            score = fields[iscore]

            g1 = protein_to_gene.get(p1)
            g2 = protein_to_gene.get(p2)

            if g1 is None:
                n_missing_p1 += 1
            if g2 is None:
                n_missing_p2 += 1

            # A gene-level association is written only if both endpoints map.
            if g1 is None or g2 is None:
                continue

            writer.writerow([g1, g2, score])
            n_mapped += 1

    print()
    print("STRING mapping summary")
    print(f"  STRING rows read:                     {n_string:,}")
    print(f"  rows with both endpoints mapped:      {n_mapped:,}")
    print(f"  rows missing protein1 mapping:        {n_missing_p1:,}")
    print(f"  rows missing protein2 mapping:        {n_missing_p2:,}")
    print(f"  output:                               {args.output}")
    print()
    print(
        "Note: score filtering and gene-pair deduplication are intentionally "
        "left to Prepare_data.py."
    )


if __name__ == "__main__":
    main()

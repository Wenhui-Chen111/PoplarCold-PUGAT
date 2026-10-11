#!/usr/bin/env python3
"""
Extract all unique protein accessions from a STRING protein.links file.

Example
-------
python extract_string_accessions.py \
    --input 3694.protein.links.v12.5.txt.gz \
    --output string_protein_accessions.txt \
    --taxon-prefix 3694.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def normalize_accession(value: str, taxon_prefix: str) -> str:
    value = value.strip()
    if taxon_prefix and value.startswith(taxon_prefix):
        value = value[len(taxon_prefix):]
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Extract unique protein1/protein2 accessions from a STRING protein.links file."
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="STRING protein.links file (.txt or .txt.gz).")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output text file, one accession per line.")
    parser.add_argument("--taxon-prefix", default="3694.",
                        help="Prefix to remove from STRING protein IDs (default: 3694.).")
    args = parser.parse_args()

    accessions = set()
    n_edges = 0

    with open_text(args.input) as fh:
        header = fh.readline().strip().split()
        required = {"protein1", "protein2", "combined_score"}
        if not required.issubset(header):
            raise ValueError(
                f"Expected STRING columns {sorted(required)}, found: {header}"
            )

        i1 = header.index("protein1")
        i2 = header.index("protein2")

        for line_no, line in enumerate(fh, start=2):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) <= max(i1, i2):
                raise ValueError(f"Malformed line {line_no}: {line.rstrip()}")

            p1 = normalize_accession(fields[i1], args.taxon_prefix)
            p2 = normalize_accession(fields[i2], args.taxon_prefix)

            if p1:
                accessions.add(p1)
            if p2:
                accessions.add(p2)
            n_edges += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wt", encoding="utf-8", newline="\n") as out:
        for accession in sorted(accessions):
            out.write(accession + "\n")

    print(f"STRING edge rows read: {n_edges:,}")
    print(f"Unique protein accessions: {len(accessions):,}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Reviewer-facing paired per-split comparison against PoplarCold-PUGAT.

This script performs *post-processing only*; it does not train any model.
It verifies that the main model and every baseline contain exactly the same
(repeat, split_seed) pairs, then writes both the individual paired results and
summary statistics needed to address the reviewer request:

    "Report paired results for every data split so that the magnitude and
     stability of the improvement can be assessed."

Recommended inputs
------------------
- main model:     <main_run>/per_split_metrics.csv
- all baselines:  <baseline_run>/all_baselines_per_split.csv

Outputs
-------
- paired_vs_main_per_split_long.csv
    One row per baseline x split x metric with main value, baseline value,
    and delta = main - baseline.
- paired_vs_main_per_split_wide.csv
    One row per baseline x split with compact columns for the selected metrics.
- paired_vs_main_summary.csv
    Mean/median delta, bootstrap 95% CI, Wilcoxon p, Holm-adjusted p, and
    counts/fractions of splits where main is higher/lower/tied.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PRIMARY_METRICS: Tuple[str, ...] = ("nDCG@100", "Recall@100", "AUPRC", "AUROC")
KEYS: Tuple[str, ...] = ("repeat", "split_seed")


def _as_int_keys(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    missing = [c for c in KEYS if c not in out.columns]
    if missing:
        raise ValueError(f"{source} is missing pairing key column(s): {missing}")
    for c in KEYS:
        out[c] = pd.to_numeric(out[c], errors="raise").astype(int)
    if out.duplicated(list(KEYS)).any():
        dup = out.loc[out.duplicated(list(KEYS), keep=False), list(KEYS)]
        raise ValueError(
            f"{source} contains duplicate (repeat, split_seed) rows; first examples:\n"
            f"{dup.head(10).to_string(index=False)}"
        )
    return out


def _split_key_set(df: pd.DataFrame) -> set:
    return set(map(tuple, df.loc[:, list(KEYS)].to_numpy().tolist()))


def _validate_exact_pairing(main: pd.DataFrame, base: pd.DataFrame, name: str) -> None:
    main_keys = _split_key_set(main)
    base_keys = _split_key_set(base)
    if main_keys != base_keys:
        only_main = sorted(main_keys - base_keys)
        only_base = sorted(base_keys - main_keys)
        raise ValueError(
            f"Baseline '{name}' does not use exactly the same split pairs as the main model. "
            f"Only in main (first 10): {only_main[:10]}; "
            f"only in baseline (first 10): {only_base[:10]}"
        )


def _bootstrap_mean_ci(values: np.ndarray, reps: int, seed: int) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(reps, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.where(np.isfinite(p))[0]
    if valid.size == 0:
        return out.tolist()
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * p[idx])
        running = max(running, adjusted)
        out[idx] = running
    return out.tolist()


def _load_baselines(all_baselines_path: str) -> List[Tuple[str, pd.DataFrame]]:
    df = pd.read_csv(all_baselines_path)
    if "method" not in df.columns:
        raise ValueError("all_baselines_per_split.csv must contain a 'method' column")
    baselines: List[Tuple[str, pd.DataFrame]] = []
    for method, sub in df.groupby("method", sort=False):
        baselines.append((str(method), _as_int_keys(sub.copy(), f"baseline {method}")))
    if not baselines:
        raise ValueError("No baseline rows found")
    return baselines


def compare(
    main_metrics_path: str,
    all_baselines_path: str,
    out_dir: str,
    metrics: Iterable[str] = PRIMARY_METRICS,
    bootstrap_reps: int = 10000,
    bootstrap_seed: int = 20261001,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    main = _as_int_keys(pd.read_csv(main_metrics_path), "main model")
    baselines = _load_baselines(all_baselines_path)
    metrics = list(metrics)

    missing_main = [m for m in metrics if m not in main.columns]
    if missing_main:
        raise ValueError(f"Main model metrics file is missing: {missing_main}")

    detailed_rows = []
    summary_rows = []

    for baseline_idx, (name, base) in enumerate(baselines):
        _validate_exact_pairing(main, base, name)
        missing_base = [m for m in metrics if m not in base.columns]
        if missing_base:
            raise ValueError(f"Baseline '{name}' is missing metric(s): {missing_base}")

        merged = main.loc[:, [*KEYS, *metrics]].merge(
            base.loc[:, [*KEYS, *metrics]],
            on=list(KEYS),
            how="inner",
            validate="one_to_one",
            suffixes=("_main", "_baseline"),
        ).sort_values(list(KEYS)).reset_index(drop=True)

        for metric_idx, metric in enumerate(metrics):
            a = pd.to_numeric(merged[f"{metric}_main"], errors="coerce").to_numpy(float)
            b = pd.to_numeric(merged[f"{metric}_baseline"], errors="coerce").to_numpy(float)
            valid = np.isfinite(a) & np.isfinite(b)
            a = a[valid]
            b = b[valid]
            paired_keys = merged.loc[valid, list(KEYS)].reset_index(drop=True)
            delta = a - b

            for i in range(len(delta)):
                if np.isclose(delta[i], 0.0, atol=1e-12, rtol=0.0):
                    direction = "tie"
                elif delta[i] > 0:
                    direction = "main_higher"
                else:
                    direction = "baseline_higher"
                detailed_rows.append({
                    "baseline": name,
                    "repeat": int(paired_keys.loc[i, "repeat"]),
                    "split_seed": int(paired_keys.loc[i, "split_seed"]),
                    "metric": metric,
                    "main_value": float(a[i]),
                    "baseline_value": float(b[i]),
                    "delta_main_minus_baseline": float(delta[i]),
                    "direction": direction,
                })

            if len(delta) == 0:
                continue
            if len(delta) < 2 or np.allclose(delta, 0.0):
                p_value = 1.0 if np.allclose(delta, 0.0) else np.nan
            else:
                try:
                    p_value = float(wilcoxon(
                        delta,
                        zero_method="wilcox",
                        alternative="two-sided",
                    ).pvalue)
                except ValueError:
                    p_value = np.nan

            ci_low, ci_high = _bootstrap_mean_ci(
                delta,
                reps=bootstrap_reps,
                seed=bootstrap_seed + baseline_idx * 100 + metric_idx,
            )
            n_main = int(np.sum(delta > 1e-12))
            n_base = int(np.sum(delta < -1e-12))
            n_tie = int(len(delta) - n_main - n_base)

            summary_rows.append({
                "baseline": name,
                "metric": metric,
                "n_pairs": int(len(delta)),
                "n_main_higher": n_main,
                "n_baseline_higher": n_base,
                "n_ties": n_tie,
                "fraction_main_higher": float(n_main / len(delta)),
                "mean_delta_main_minus_baseline": float(np.mean(delta)),
                "median_delta_main_minus_baseline": float(np.median(delta)),
                "min_delta_main_minus_baseline": float(np.min(delta)),
                "max_delta_main_minus_baseline": float(np.max(delta)),
                "delta_ci95_low": ci_low,
                "delta_ci95_high": ci_high,
                "wilcoxon_p": p_value,
            })

    detail = pd.DataFrame(detailed_rows)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["holm_adjusted_p"] = _holm_adjust(summary["wilcoxon_p"].tolist())

    detail_path = out / "paired_vs_main_per_split_long.csv"
    summary_path = out / "paired_vs_main_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    if detail.empty:
        wide = pd.DataFrame()
    else:
        value_wide = detail.pivot(
            index=["baseline", "repeat", "split_seed"],
            columns="metric",
            values=["main_value", "baseline_value", "delta_main_minus_baseline"],
        )
        value_wide.columns = [f"{kind}__{metric}" for kind, metric in value_wide.columns]
        wide = value_wide.reset_index()
    wide_path = out / "paired_vs_main_per_split_wide.csv"
    wide.to_csv(wide_path, index=False)

    print(f"[done] Detailed per-split pairs: {detail_path}")
    print(f"[done] Wide per-split table:     {wide_path}")
    print(f"[done] Paired summary:           {summary_path}")
    return detail, wide, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create reviewer-facing paired per-split main-vs-baseline result tables"
    )
    parser.add_argument("--main_metrics", required=True,
                        help="PoplarCold-PUGAT per_split_metrics.csv")
    parser.add_argument("--all_baselines", required=True,
                        help="all_baselines_per_split.csv from run_all_baselines.py")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--metrics", nargs="+", default=list(PRIMARY_METRICS),
                        help="Metrics to pair; default: nDCG@100 Recall@100 AUPRC AUROC")
    parser.add_argument("--bootstrap_reps", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20261001)
    args = parser.parse_args()

    compare(
        main_metrics_path=args.main_metrics,
        all_baselines_path=args.all_baselines,
        out_dir=args.out_dir,
        metrics=args.metrics,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()

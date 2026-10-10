# -*- coding: utf-8 -*-
"""Run only the five baseline methods named in the manuscript.

Outputs are written to one subdirectory per method plus combined paired per-split CSVs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baseline_common import load_preprocessed_data, load_split_manifest, run_repeated_baseline, summarize_per_split
from baseline_xgboost import METHOD_NAME as XGB_NAME, METHOD_SPEC as XGB_SPEC, fit_score as fit_xgb
from baseline_pu_bagging import METHOD_NAME as PUB_NAME, METHOD_SPEC as PUB_SPEC, fit_score as fit_pub
from baseline_label_propagation import METHOD_NAME as LP_NAME, METHOD_SPEC as LP_SPEC, fit_score as fit_lp
from baseline_emogi_variant import METHOD_NAME as EMOGI_NAME, METHOD_SPEC as EMOGI_SPEC, fit_score as fit_emogi
from baseline_rgcn import METHOD_NAME as RGCN_NAME, METHOD_SPEC as RGCN_SPEC, fit_score as fit_rgcn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the five manuscript baselines on one fixed split manifest")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--split_mode", required=True, choices=["random", "grouped"])
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--train_seed_base", type=int, default=10042)
    parser.add_argument("--max_trials", type=int, default=12,
                        help="Default tuning cap for trainable baselines")
    parser.add_argument("--tune_epochs", type=int, default=200)
    parser.add_argument("--final_epochs", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--pi", type=float, default=0.013)
    parser.add_argument("--xgb_estimators", type=int, default=500)
    parser.add_argument("--pu_bags", type=int, default=100)
    parser.add_argument("--lp_max_iter", type=int, default=1000)
    parser.add_argument("--no_save_scores", action="store_true")
    args = parser.parse_args()

    data = load_preprocessed_data(args.data_dir)
    manifest = load_split_manifest(args.split_manifest, data, args.split_mode)
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    save_scores = not args.no_save_scores

    jobs = [
        (XGB_NAME, XGB_SPEC, lambda d, m, s: fit_xgb(
            d, m, s, max_trials=args.max_trials, n_estimators=args.xgb_estimators
        )),
        (PUB_NAME, PUB_SPEC, lambda d, m, s: fit_pub(
            d, m, s, max_trials=args.max_trials, n_bags=args.pu_bags
        )),
        (LP_NAME, LP_SPEC, lambda d, m, s: fit_lp(
            d, m, s, max_iter=args.lp_max_iter
        )),
        (EMOGI_NAME, EMOGI_SPEC, lambda d, m, s: fit_emogi(
            d, m, s,
            max_trials=args.max_trials,
            tune_epochs=args.tune_epochs,
            final_epochs=args.final_epochs,
            eval_every=args.eval_every,
        )),
        (RGCN_NAME, RGCN_SPEC, lambda d, m, s: fit_rgcn(
            d, m, s,
            pi=args.pi,
            max_trials=args.max_trials,
            tune_epochs=args.tune_epochs,
            final_epochs=args.final_epochs,
            eval_every=args.eval_every,
        )),
    ]

    combined = []
    for method_name, method_spec, fit_fn in jobs:
        safe_name = method_name.lower().replace("-", "_").replace(" ", "_")
        print("\n" + "=" * 100)
        print(f"Running {method_name}")
        print("=" * 100)
        metrics_df, _ = run_repeated_baseline(
            data=data,
            manifest=manifest,
            out_dir=str(root / safe_name),
            method_name=method_name,
            method_spec=method_spec,
            fit_score_fn=fit_fn,
            train_seed_base=args.train_seed_base,
            save_scores=save_scores,
        )
        combined.append(metrics_df)

    all_df = pd.concat(combined, ignore_index=True)
    all_df.to_csv(root / "all_baselines_per_split.csv", index=False)

    metric_cols = [c for c in all_df.columns if c.startswith(("nDCG@", "Recall@", "Precision@", "Hits@"))]
    metric_cols += [c for c in ["MRR", "AUPRC", "AUROC"] if c in all_df.columns]
    summary_frames = []
    for method, sub in all_df.groupby("method", sort=False):
        s = summarize_per_split(sub, metric_cols)
        s.insert(0, "method", method)
        summary_frames.append(s)
    pd.concat(summary_frames, ignore_index=True).to_csv(root / "all_baselines_summary.csv", index=False)

    # Wide paired table: one row per repeat, useful for paired tests/rebuttal supplements.
    key_metrics = [m for m in ["nDCG@100", "Recall@100", "AUPRC", "AUROC"] if m in all_df.columns]
    wide = all_df.pivot(index=["repeat", "split_seed"], columns="method", values=key_metrics)
    wide.columns = [f"{metric}__{method}" for metric, method in wide.columns]
    wide.reset_index().to_csv(root / "paired_per_split_wide.csv", index=False)

    print(f"\n[done] Combined outputs: {root}")


if __name__ == "__main__":
    main()

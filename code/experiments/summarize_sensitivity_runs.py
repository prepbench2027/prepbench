#!/usr/bin/env python3
"""
Summarize PrepBench sensitivity runs.

Use this after running the zero-penalty and/or stronger-EA configurations. It
combines experiment_records.csv files, reports per-config scores/evaluations,
paired differences against D-Rand and D-S-F-A, and writes compact LaTeX tables.

Example
-------
python experiments/summarize_sensitivity_runs.py \
  --runs main300=results/eab300 main600=results/eab600 \
         zpen300=results/zpen300 zpen600=results/zpen600 \
         strong300=results/strong300 strong600=results/strong600 \
  --out results/sensitivity_summary
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def parse_run_arg(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError("run spec must be label=path")
    label, path = arg.split("=", 1)
    return label, Path(path)


def load_records(run_specs: list[tuple[str, Path]]) -> pd.DataFrame:
    frames = []
    for label, path in run_specs:
        fp = path / "experiment_records.csv"
        if not fp.exists():
            raise FileNotFoundError(fp)
        df = pd.read_csv(fp)
        df["run_label"] = label
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for col in ["budget", "seed", "score", "n_full_evals", "n_ok_evals", "n_screened_out", "n_pruned", "nodes", "branches", "depth", "fitness_cost_penalty", "fitness_complexity_penalty"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def per_config_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["run_label", "budget", "config"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            n_runs=("score", "size"),
            n_datasets=("dataset", "nunique"),
            n_seeds=("seed", "nunique"),
            mean_full_evals=("n_full_evals", "mean"),
            mean_ok_evals=("n_ok_evals", "mean"),
            mean_screened=("n_screened_out", "mean"),
            mean_pruned=("n_pruned", "mean"),
            mean_nodes=("nodes", "mean"),
            mean_branches=("branches", "mean"),
            mean_depth=("depth", "mean"),
            cost_penalty=("fitness_cost_penalty", "first"),
            complexity_penalty=("fitness_complexity_penalty", "first"),
        )
        .sort_values(["budget", "run_label", "mean_score"], ascending=[True, True, False])
    )


def paired_tests(df: pd.DataFrame, baselines: list[str]) -> pd.DataFrame:
    # Average seeds first; dataset is the inference unit.
    avg = (
        df.groupby(["run_label", "budget", "dataset", "config"], as_index=False)
        .agg(score=("score", "mean"))
    )
    rows = []
    for (label, budget), sub in avg.groupby(["run_label", "budget"]):
        pivot = sub.pivot(index="dataset", columns="config", values="score")
        for base in baselines:
            if base not in pivot:
                continue
            for cfg in pivot.columns:
                if cfg == base:
                    continue
                diff = pivot[cfg] - pivot[base]
                vals = diff.dropna()
                if len(vals) < 2:
                    p = np.nan
                    stat = np.nan
                else:
                    try:
                        stat, p = wilcoxon(vals.values, zero_method="wilcox")
                    except ValueError:
                        stat, p = np.nan, np.nan
                rows.append(
                    {
                        "run_label": label,
                        "budget": budget,
                        "config": cfg,
                        "baseline": base,
                        "n_datasets": len(vals),
                        "mean_diff": vals.mean(),
                        "median_diff": vals.median(),
                        "wins": int((vals > 0).sum()),
                        "ties": int((vals == 0).sum()),
                        "losses": int((vals < 0).sum()),
                        "wilcoxon_stat": stat,
                        "wilcoxon_p": p,
                    }
                )
    return pd.DataFrame(rows).sort_values(["budget", "run_label", "baseline", "config"])


def fairness_from_records(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    status_cols = [c for c in ["n_full_evals", "n_ok_evals", "n_screened_out", "n_pruned"] if c in df.columns]
    agg = {
        "score": "mean",
        "nodes": "mean",
        "branches": "mean",
        "depth": "mean",
    }
    for c in status_cols:
        agg[c] = "mean"
    out = df.groupby(["run_label", "budget", "config"], as_index=False).agg(agg)
    out = out.rename(columns={
        "score": "mean_score",
        "nodes": "mean_nodes",
        "branches": "mean_branches",
        "depth": "mean_depth",
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=parse_run_arg, required=True, help="label=path entries")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baselines", nargs="+", default=["D-Rand", "D-S-F-A"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = load_records(args.runs)
    records.to_csv(out / "combined_sensitivity_records.csv", index=False)

    summary = per_config_summary(records)
    summary.to_csv(out / "sensitivity_summary_by_config.csv", index=False)

    pairs = paired_tests(records, args.baselines)
    pairs.to_csv(out / "sensitivity_paired_tests.csv", index=False)

    fair = fairness_from_records(records)
    fair.to_csv(out / "fairness_audit_from_records.csv", index=False)

    tex = out / "latex"
    tex.mkdir(exist_ok=True)
    keep = [c for c in ["run_label", "budget", "config", "mean_score", "mean_full_evals", "mean_nodes", "mean_branches", "mean_depth", "cost_penalty", "complexity_penalty"] if c in summary.columns]
    summary[keep].round(4).to_latex(tex / "sensitivity_summary_by_config.tex", index=False, escape=True)

    keep_pairs = [c for c in ["run_label", "budget", "config", "baseline", "mean_diff", "median_diff", "wins", "ties", "losses", "wilcoxon_p"] if c in pairs.columns]
    pairs[keep_pairs].round(5).to_latex(tex / "sensitivity_paired_tests.tex", index=False, escape=True)

    print(f"Wrote {out.resolve()}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()


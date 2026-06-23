#!/usr/bin/env python3
"""
VLDB EAB add-on analyses for PrepBench.

This script turns the benchmark outputs into the extra evidence requested by an
EAB review: coverage/provenance audit, random-baseline fairness audit, dataset
stress/subgroup analysis, anytime/incumbent behavior, workflow diversity, and
TOST margin sensitivity.

Inputs
------
One or more result directories. Each directory should contain at least
``experiment_records.csv`` and, for trace-level analyses, ``search_history.csv``.

Example
-------
python experiments/analyze_extras.py \
  --run-dirs results_60s results_300s results_600s \
  --out extra_analysis \
  --seeds 7 11 23 29 41
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from scipy.stats import ttest_1samp
except Exception:  # pragma: no cover
    ttest_1samp = None

try:
    from data_loader.load_deepline import ALL_DATASET_NAMES, load_deepline_dataset
except Exception:  # pragma: no cover
    ALL_DATASET_NAMES = []
    load_deepline_dataset = None

try:
    from parallel_evolutionary_pipeline.search_earlystop import RUN_MATRIX
except Exception:  # pragma: no cover
    RUN_MATRIX = {}


RANDOM_CONFIGS = ["D-Rand", "Flat-Rand"]
PRIMARY_BASELINES = ["D-Rand", "Flat-Rand"]
CONFIG_ORDER = list(RUN_MATRIX.keys()) if RUN_MATRIX else [
    "L-S-F-A", "B-S-F-A", "D-S-F-A", "D-C-F-A", "D-S-F-B",
    "D-S-T-A", "D-C-T-B", "D-Rand", "Flat-Rand",
]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, on_bad_lines="skip")


def _infer_budget_from_dir(path: Path) -> float | None:
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*s", path.name.lower())
    if m:
        return float(m.group(1))
    m = re.search(r"budget[_-]?(\d+(?:\.\d+)?)", path.name.lower())
    if m:
        return float(m.group(1))
    return None


def load_runs(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, histories = [], []
    for rd in run_dirs:
        rec = _safe_read_csv(rd / "experiment_records.csv")
        hist = _safe_read_csv(rd / "search_history.csv")
        inferred_budget = _infer_budget_from_dir(rd)
        for df in (rec, hist):
            if df.empty:
                continue
            df["source_dir"] = str(rd)
            if "budget" not in df.columns and inferred_budget is not None:
                df["budget"] = inferred_budget
        if not rec.empty:
            records.append(rec)
        if not hist.empty:
            histories.append(hist)
    rec = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    hist = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    # Normalize numeric dtypes used below.
    for df in (rec, hist):
        if df.empty:
            continue
        for col in ["budget", "seed", "score", "elapsed_seconds", "best_so_far", "branches", "nodes", "depth", "runtime_seconds", "estimated_cost", "serial_cost"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return rec, hist


def entropy(values: Iterable[object]) -> float:
    vals = [str(v) for v in values if pd.notna(v)]
    if not vals:
        return float("nan")
    c = Counter(vals)
    n = sum(c.values())
    return float(-sum((v / n) * math.log(v / n, 2) for v in c.values()))


def workflow_signature_from_row(row: pd.Series) -> str:
    if "graph_signature" in row and pd.notna(row["graph_signature"]):
        return str(row["graph_signature"])
    # Backward-compatible approximation for older traces. This is weaker than
    # the patched trace but still useful for diagnosing duplicate patterns.
    parts = []
    for col in ["branches", "nodes", "depth", "operator", "estimator", "branch_ops", "tail_ops"]:
        if col in row and pd.notna(row[col]):
            parts.append(f"{col}={row[col]}")
    return "|".join(parts) if parts else "unknown"


def build_dataset_metadata(records: pd.DataFrame) -> pd.DataFrame:
    names = sorted(records["dataset"].dropna().astype(str).unique()) if not records.empty else sorted(ALL_DATASET_NAMES)
    rows = []
    for name in names:
        if load_deepline_dataset is None:
            continue
        try:
            ds = load_deepline_dataset(name)
            y = ds.y.astype("category")
            counts = y.value_counts(dropna=False)
            majority = float(counts.max() / counts.sum()) if counts.sum() else float("nan")
            missing = float(ds.X.isna().sum().sum() / max(1, ds.X.shape[0] * ds.X.shape[1]))
            rows.append({
                "dataset": name,
                "n_samples": ds.n_samples,
                "n_features": ds.n_features,
                "n_numeric": len(ds.numeric_columns),
                "n_categorical": len(ds.categorical_columns),
                "categorical_fraction": len(ds.categorical_columns) / max(1, ds.n_features),
                "missing_fraction": missing,
                "n_classes": int(y.nunique(dropna=False)),
                "majority_class_fraction": majority,
                "imbalance_gap": majority - 1.0 / max(1, int(y.nunique(dropna=False))),
            })
        except Exception as exc:
            rows.append({"dataset": name, "metadata_error": repr(exc)})
    meta = pd.DataFrame(rows)
    if meta.empty:
        return meta
    # Robust bins; duplicates='drop' handles small smoke-test subsets.
    for col, label in [
        ("n_samples", "rows_bin"),
        ("n_features", "features_bin"),
        ("categorical_fraction", "catfrac_bin"),
        ("missing_fraction", "missing_bin"),
        ("imbalance_gap", "imbalance_bin"),
    ]:
        if col in meta.columns and meta[col].notna().sum() >= 3:
            try:
                meta[label] = pd.qcut(meta[col], q=3, labels=["low", "mid", "high"], duplicates="drop")
            except Exception:
                meta[label] = "all"
        else:
            meta[label] = "all"
    if "n_classes" in meta.columns:
        meta["task_type"] = np.where(meta["n_classes"].fillna(0).astype(float) <= 2, "binary", "multiclass")
    return meta


def coverage_audit(records: pd.DataFrame, expected_seeds: list[int] | None) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    datasets = sorted(records["dataset"].dropna().astype(str).unique())
    configs = [c for c in CONFIG_ORDER if c in set(records["config"].astype(str))]
    configs += sorted(set(records["config"].astype(str)) - set(configs))
    budgets = sorted(records["budget"].dropna().unique()) if "budget" in records else [None]
    seeds = expected_seeds or sorted(records["seed"].dropna().astype(int).unique())
    got = set(zip(records["dataset"].astype(str), records["config"].astype(str), records["seed"].astype(int), records["budget"].astype(float)))
    rows = []
    for b in budgets:
        for cfg in configs:
            expected = len(datasets) * len(seeds)
            observed = int(((records["budget"] == b) & (records["config"].astype(str) == cfg)).sum())
            missing = []
            for ds in datasets:
                for seed in seeds:
                    key = (ds, cfg, int(seed), float(b))
                    if key not in got:
                        missing.append(f"{ds}:seed{seed}")
            rows.append({
                "budget": b, "config": cfg, "expected_runs": expected,
                "observed_runs": observed, "missing_runs": expected - observed,
                "missing_examples": ";".join(missing[:20]),
            })
    return pd.DataFrame(rows)


def random_fairness_audit(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    h = history.copy()
    if "graph_signature" not in h.columns:
        h["graph_signature"] = h.apply(workflow_signature_from_row, axis=1)
    if "depth" not in h.columns:
        h["depth"] = np.nan
    status_col = h["status"].astype(str) if "status" in h.columns else pd.Series("unknown", index=h.index)
    h["full_eval"] = ~status_col.isin(["screened_out", "pruned"])
    h["ok_eval"] = status_col.eq("ok")
    rows = []
    group_cols = ["budget", "config"] if "budget" in h.columns else ["config"]
    for key, sub in h.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(group_cols, key))
        n = len(sub)
        unique = sub["graph_signature"].nunique(dropna=True)
        record.update({
            "n_candidates": n,
            "unique_workflows": int(unique),
            "duplicate_rate": float(1.0 - unique / n) if n else np.nan,
            "mean_nodes": sub["nodes"].mean() if "nodes" in sub else np.nan,
            "mean_branches": sub["branches"].mean() if "branches" in sub else np.nan,
            "mean_depth": sub["depth"].mean() if "depth" in sub else np.nan,
            "full_eval_rate": sub["full_eval"].mean(),
            "ok_eval_rate": sub["ok_eval"].mean(),
            "operator_entropy": entropy(sub["operator"]) if "operator" in sub else np.nan,
            "workflow_entropy": entropy(sub["graph_signature"]),
            "estimator_entropy": entropy(sub["estimator"]) if "estimator" in sub else np.nan,
            "mean_estimated_cost": sub["estimated_cost"].mean() if "estimated_cost" in sub else np.nan,
            "mean_runtime_seconds": sub["runtime_seconds"].mean() if "runtime_seconds" in sub else np.nan,
        })
        rows.append(record)
    return pd.DataFrame(rows).sort_values(group_cols)


def incumbent_analysis(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [c for c in ["budget", "dataset", "config", "seed"] if c in history.columns]
    for key, sub in history.sort_values("elapsed_seconds").groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rec = dict(zip(group_cols, key))
        valid = sub.dropna(subset=["best_so_far", "elapsed_seconds"])
        scored = sub.dropna(subset=["score", "elapsed_seconds"])
        if valid.empty:
            rec.update({"time_to_first_valid": np.nan, "time_to_best": np.nan, "last_improvement_time": np.nan, "incumbent_updates": 0, "final_best": np.nan, "n_candidates": len(sub)})
        else:
            final_best = valid["best_so_far"].iloc[-1]
            best_rows = valid.loc[valid["best_so_far"] >= final_best]
            improved = sub["improved_incumbent"].fillna(False).astype(bool) if "improved_incumbent" in sub.columns else valid["best_so_far"].diff().fillna(1).gt(0)
            rec.update({
                "time_to_first_valid": scored["elapsed_seconds"].iloc[0] if not scored.empty else np.nan,
                "time_to_best": best_rows["elapsed_seconds"].iloc[0] if not best_rows.empty else np.nan,
                "last_improvement_time": sub.loc[improved, "elapsed_seconds"].max() if improved.any() else np.nan,
                "incumbent_updates": int(improved.sum()),
                "final_best": final_best,
                "n_candidates": len(sub),
            })
        rows.append(rec)
    return pd.DataFrame(rows)


def subgroup_analysis(records: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if records.empty or meta.empty:
        return pd.DataFrame()
    df = records.merge(meta, on="dataset", how="left")
    # Aggregate seeds first so each dataset contributes one unit per config/budget.
    base = df.groupby(["budget", "dataset", "config"], as_index=False).agg(score=("score", "mean"))
    base = base.merge(meta, on="dataset", how="left")
    rows = []
    bin_cols = [c for c in ["rows_bin", "features_bin", "catfrac_bin", "missing_bin", "imbalance_bin", "task_type"] if c in base.columns]
    for bcol in bin_cols:
        for (budget, bin_value, config), sub in base.groupby(["budget", bcol, "config"], dropna=False):
            rows.append({
                "budget": budget,
                "subgroup_axis": bcol,
                "subgroup": str(bin_value),
                "config": config,
                "n_dataset_config_points": len(sub),
                "mean_score": sub["score"].mean(),
                "median_score": sub["score"].median(),
            })
    return pd.DataFrame(rows)


def tost_paired(diffs: pd.Series, margin: float) -> tuple[float, bool]:
    vals = diffs.dropna().astype(float)
    if len(vals) < 2:
        return np.nan, False
    if ttest_1samp is None:
        # Fallback: conservative CI check using normal approximation.
        se = vals.std(ddof=1) / math.sqrt(len(vals))
        if se == 0:
            return 0.0 if abs(vals.mean()) < margin else 1.0, abs(vals.mean()) < margin
        lo = vals.mean() - 1.96 * se
        hi = vals.mean() + 1.96 * se
        return np.nan, bool(lo > -margin and hi < margin)
    # TOST: H01 mean <= -margin and H02 mean >= margin.
    t_low = ttest_1samp(vals + margin, popmean=0.0, alternative="greater")
    t_high = ttest_1samp(vals - margin, popmean=0.0, alternative="less")
    p = max(float(t_low.pvalue), float(t_high.pvalue))
    return p, p < 0.05


def tost_sensitivity(records: pd.DataFrame, margins: list[float]) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    # Seed average first, then paired by dataset.
    avg = records.groupby(["budget", "dataset", "config"], as_index=False).agg(score=("score", "mean"))
    rows = []
    for budget, sub in avg.groupby("budget"):
        pivot = sub.pivot(index="dataset", columns="config", values="score")
        for baseline in PRIMARY_BASELINES:
            if baseline not in pivot.columns:
                continue
            for cfg in pivot.columns:
                if cfg == baseline:
                    continue
                diffs = pivot[cfg] - pivot[baseline]
                for margin in margins:
                    p, equiv = tost_paired(diffs, margin)
                    rows.append({
                        "budget": budget,
                        "baseline": baseline,
                        "config": cfg,
                        "margin": margin,
                        "n_pairs": int(diffs.dropna().shape[0]),
                        "mean_diff": diffs.mean(),
                        "median_diff": diffs.median(),
                        "tost_p_value": p,
                        "equivalent_at_0_05": equiv,
                    })
    return pd.DataFrame(rows)


def write_latex_tables(out: Path, fairness: pd.DataFrame, inc: pd.DataFrame, tost: pd.DataFrame) -> None:
    texdir = out / "latex"
    texdir.mkdir(parents=True, exist_ok=True)
    if not fairness.empty:
        cols = [c for c in ["budget", "config", "n_candidates", "unique_workflows", "duplicate_rate", "mean_nodes", "mean_branches", "operator_entropy", "workflow_entropy"] if c in fairness.columns]
        fairness[cols].round(4).to_latex(texdir / "random_fairness_audit.tex", index=False, escape=True)
    if not inc.empty:
        summ = inc.groupby(["budget", "config"], as_index=False).agg(
            mean_time_to_first_valid=("time_to_first_valid", "mean"),
            mean_time_to_best=("time_to_best", "mean"),
            mean_last_improvement_time=("last_improvement_time", "mean"),
            mean_incumbent_updates=("incumbent_updates", "mean"),
        )
        summ.round(4).to_latex(texdir / "incumbent_summary.tex", index=False, escape=True)
    if not tost.empty:
        tost.round(5).to_latex(texdir / "tost_sensitivity.tex", index=False, escape=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True, help="directories containing experiment_records.csv and search_history.csv")
    ap.add_argument("--out", default="extra_analysis")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="expected seeds for coverage audit")
    ap.add_argument("--margins", nargs="+", type=float, default=[0.005, 0.01, 0.02])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dirs = [Path(p) for p in args.run_dirs]
    records, history = load_runs(run_dirs)

    if records.empty:
        raise SystemExit("No experiment_records.csv found in --run-dirs.")

    records.to_csv(out / "combined_experiment_records.csv", index=False)
    if not history.empty:
        history.to_csv(out / "combined_search_history.csv", index=False)

    meta = build_dataset_metadata(records)
    meta.to_csv(out / "dataset_metadata.csv", index=False)

    cov = coverage_audit(records, args.seeds)
    cov.to_csv(out / "coverage_audit.csv", index=False)

    fair = random_fairness_audit(history)
    fair.to_csv(out / "random_fairness_audit.csv", index=False)

    inc = incumbent_analysis(history)
    inc.to_csv(out / "incumbent_update_analysis.csv", index=False)

    sub = subgroup_analysis(records, meta)
    sub.to_csv(out / "subgroup_stress_analysis.csv", index=False)

    tost = tost_sensitivity(records, args.margins)
    tost.to_csv(out / "tost_margin_sensitivity.csv", index=False)

    write_latex_tables(out, fair, inc, tost)

    manifest = {
        "inputs": [str(p) for p in run_dirs],
        "outputs": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()),
        "notes": [
            "coverage_audit.csv must show zero missing_runs before final paper tables are regenerated.",
            "random_fairness_audit.csv is strongest when search_history.csv contains graph_signature, branch_ops, tail_ops, estimator, depth, and cost fields from the patched tracer.",
            "TOST is paired by dataset after averaging seeds; report margin sensitivity for 0.005/0.01/0.02.",
        ],
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run the ICDE EAB PrepBench matrix with resume support and complete trace output.

This is a cleaned-up replacement for experiments/run_matrix_deepline_parallel_es.py.
It keeps the same CSV schema but adds stronger filtering, a coverage report, and
is designed to be used with the patched search_earlystop.py that records rich
candidate-level traces.

Examples
--------
# Full 56-dataset / 9-config / 5-seed run at 300 seconds.
python experiments/run_matrix.py \
  --budget 300 \
  --seeds 7 11 23 29 41 \
  --out results_300s \
  --processes 16

# Long-budget confirmation on only the most important configs.
python experiments/run_matrix.py \
  --budget 1200 \
  --configs D-Rand Flat-Rand D-S-F-A D-C-T-B \
  --seeds 7 11 23 29 41 \
  --out results_1200s_key_configs \
  --processes 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import multiprocessing.pool as mpp
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv()

from parallel_evolutionary_pipeline.config import SearchConfig
from parallel_evolutionary_pipeline.graph import WorkflowIndividual
from parallel_evolutionary_pipeline.search_earlystop import RUN_MATRIX, ParallelAwareEA
from data_loader.load_deepline import ALL_DATASET_NAMES, load_deepline_dataset

warnings.filterwarnings("ignore")
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
file_lock = None


# ── Non-daemonic pool (workers may spawn subprocesses) ─────────────────
# The default mp.Pool creates daemonic workers that cannot spawn children.
# We need non-daemonic workers because each evaluation inside ParallelAwareEA
# may spawn a hard-timeout subprocess (evaluate_graph_cv/_screen).
#
# The key trick: inherit from the spawn-context Process class (which has the
# correct _Popen classmethod), rather than from bare mp.Process, and create
# a *fresh context subclass* so we never mutate the global spawn-context
# singleton.


_SpawnProcess = mp.get_context("spawn").Process


class NoDaemonProcess(_SpawnProcess):
    """A Process that is never daemonic, so it can spawn children."""

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class _NoDaemonSpawnContext(type(mp.get_context("spawn"))):
    """A spawn context that uses NoDaemonProcess instead of the default."""
    Process = NoDaemonProcess


class NoDaemonPool(mpp.Pool):
    """Pool whose workers can spawn child processes."""

    def __init__(self, *args, **kwargs):
        kwargs["context"] = _NoDaemonSpawnContext()
        super().__init__(*args, **kwargs)


RANDOM_CONFIGS = ["D-Rand", "Flat-Rand"]
PRIMARY_BASELINES = ["D-Rand", "Flat-Rand"]
CONFIG_ORDER = list(RUN_MATRIX.keys()) if RUN_MATRIX else [
    "L-S-F-A", "B-S-F-A", "D-S-F-A", "D-C-F-A", "D-S-F-B",
    "D-S-T-A", "D-C-T-B", "D-Rand", "Flat-Rand",
]

def init_worker(lock: mp.Lock) -> None:
    global file_lock
    file_lock = lock


def _load_or_skip(ds_name: str) -> Any:
    try:
        ds = load_deepline_dataset(ds_name)
        print(f"[dataset] {ds_name}: n={ds.n_samples} d={ds.n_features} "
              f"(num={len(ds.numeric_columns)} cat={len(ds.categorical_columns)})", flush=True)
        return ds
    except Exception as exc:
        print(f"[warn] could not load '{ds_name}' ({type(exc).__name__}: {exc}); skipping", flush=True)
        return None


def _time_to_target(hist: pd.DataFrame, target: float) -> float:
    if hist.empty or "best_so_far" not in hist or "elapsed_seconds" not in hist:
        return float("nan")
    hit = hist.loc[pd.to_numeric(hist["best_so_far"], errors="coerce") >= target, "elapsed_seconds"]
    return float(hit.iloc[0]) if len(hit) else float("nan")


def _anytime_auc(hist: pd.DataFrame, budget: float, denom: float) -> float:
    if hist.empty or denom <= 0 or "elapsed_seconds" not in hist or "best_so_far" not in hist:
        return float("nan")
    h = hist.dropna(subset=["elapsed_seconds", "best_so_far"]).sort_values("elapsed_seconds")
    if h.empty:
        return float("nan")
    t = h["elapsed_seconds"].clip(upper=budget).to_numpy(dtype=float)
    b = h["best_so_far"].to_numpy(dtype=float)
    if len(t) == 0:
        return float("nan")
    # Ensure the curve starts at t=0 and extends to the budget.
    t = np.concatenate([[0.0], t, [budget]])
    b = np.concatenate([[b[0]], b, [b[-1]]])
    return float(_trapz(b, t) / budget / denom)


def load_completed_runs(rec_file: Path) -> set[tuple[str, str, int, float]]:
    if not rec_file.exists() or rec_file.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(rec_file, on_bad_lines="skip")
        required = {"dataset", "config", "seed", "budget"}
        if not required.issubset(df.columns):
            return set()
        return set((str(r.dataset), str(r.config), int(r.seed), float(r.budget)) for r in df.itertuples(index=False))
    except Exception:
        return set()


def append_csv_safely(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    with file_lock:
        df.to_csv(path, mode="a", index=False, header=not path.exists())


def process_one_setting(ds_name: str, cfg_id: str, seed: int, budget: float,
                        base: SearchConfig, out_dir: Path) -> tuple[dict | None, list[dict]]:
    dataset = _load_or_skip(ds_name)
    if dataset is None:
        return None, []

    conf = replace(base, random_state=seed, wall_clock_budget_seconds=budget)
    WorkflowIndividual.COST_PENALTY = conf.fitness_cost_penalty
    WorkflowIndividual.COMPLEXITY_PENALTY = conf.fitness_complexity_penalty
    method = RUN_MATRIX[cfg_id]
    result = ParallelAwareEA(conf, method).fit(dataset)
    best = result.best_individual
    history = []
    for idx, row in enumerate(result.history):
        history.append({
            "dataset": ds_name,
            "config": cfg_id,
            "seed": seed,
            "budget": budget,
            "candidate_index": idx,
            **row,
        })

    rh = pd.DataFrame(history)
    n_full = int((rh["status"].astype(str) != "screened_out").sum()) if not rh.empty and "status" in rh else 0
    n_screen = int((rh["status"].astype(str) == "screened_out").sum()) if not rh.empty and "status" in rh else 0
    n_ok = int((rh["status"].astype(str) == "ok").sum()) if not rh.empty and "status" in rh else 0

    record = dict(
        dataset=ds_name,
        config=cfg_id,
        seed=seed,
        budget=budget,
        score=best.full_score,
        estimated_cost=best.estimated_cost,
        serial_cost=best.serial_cost,
        runtime_seconds=best.runtime_seconds,
        branches=best.graph.branch_count(),
        nodes=best.graph.preprocessing_nodes(),
        depth=best.graph.depth(),
        n_full_evals=n_full,
        n_ok_evals=n_ok,
        n_screened_out=n_screen,
        n_pruned=result.pruned,
        fitness_cost_penalty=conf.fitness_cost_penalty,
        fitness_complexity_penalty=conf.fitness_complexity_penalty,
    )

    append_csv_safely(out_dir / "experiment_records.csv", [record])
    append_csv_safely(out_dir / "search_history.csv", history)

    score = best.full_score if best.full_score is not None else float("nan")
    print(f"[{ds_name}] {cfg_id:10s} seed={seed:<3} budget={budget:<6.0f} "
          f"score={score:.4f} ok={n_ok:3d} full={n_full:3d} screen={n_screen:3d} pruned={result.pruned:3d}", flush=True)
    return record, history


def write_summaries(out: Path) -> None:
    rec_file = out / "experiment_records.csv"
    hist_file = out / "search_history.csv"
    rec = pd.read_csv(rec_file, on_bad_lines="skip") if rec_file.exists() and rec_file.stat().st_size > 0 else pd.DataFrame()
    hist = pd.read_csv(hist_file, on_bad_lines="skip") if hist_file.exists() and hist_file.stat().st_size > 0 else pd.DataFrame()
    if rec.empty:
        return
    for df in (rec, hist):
        if df.empty:
            continue
        for col in [
            "budget", "seed", "score", "best_so_far", "elapsed_seconds",
            "n_full_evals", "n_ok_evals", "n_screened_out", "n_pruned",
            "branches", "nodes", "depth", "runtime_seconds",
            "estimated_cost", "serial_cost",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    targets = (0.95 * rec.groupby(["budget", "dataset"])["score"].max()).to_dict()
    best_obs = rec.groupby(["budget", "dataset"])["score"].max().to_dict()
    # Use a sorted MultiIndex on hist so each rec row lookup is O(log N)
    # instead of the O(N × len(rec)) full-scan loop.
    if hist.empty:
        met = rec[["dataset", "config", "seed", "budget", "score",
                    "n_full_evals", "n_ok_evals", "n_screened_out",
                    "n_pruned", "branches", "nodes", "depth"]].copy()
        met["time_to_target"] = float("nan")
        met["anytime_auc"] = float("nan")
    else:
        hist_idx = hist.set_index(["dataset", "config", "seed", "budget"]).sort_index()
        rows = []
        for r in rec.itertuples(index=False):
            key = (r.dataset, r.config, int(r.seed), float(r.budget))
            try:
                h = hist_idx.loc[[key]]
            except KeyError:
                h = pd.DataFrame()
            rows.append(dict(
                dataset=r.dataset,
                config=r.config,
                seed=r.seed,
                budget=r.budget,
                score=r.score,
                n_full_evals=getattr(r, "n_full_evals", np.nan),
                n_ok_evals=getattr(r, "n_ok_evals", np.nan),
                n_screened_out=getattr(r, "n_screened_out", np.nan),
                n_pruned=getattr(r, "n_pruned", np.nan),
                branches=getattr(r, "branches", np.nan),
                nodes=getattr(r, "nodes", np.nan),
                depth=getattr(r, "depth", np.nan),
                time_to_target=_time_to_target(h, targets.get((r.budget, r.dataset), float("nan"))),
                anytime_auc=_anytime_auc(h, r.budget, best_obs.get((r.budget, r.dataset), 0.0)),
            ))
        met = pd.DataFrame(rows)
    met.to_csv(out / "run_metrics.csv", index=False)
    summary = met.groupby(["budget", "config"], as_index=False).agg(
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        std_score=("score", "std"),
        mean_evals=("n_full_evals", "mean"),
        mean_ok_evals=("n_ok_evals", "mean"),
        mean_screened_out=("n_screened_out", "mean"),
        mean_pruned=("n_pruned", "mean"),
        mean_branches=("branches", "mean"),
        mean_nodes=("nodes", "mean"),
        mean_depth=("depth", "mean"),
        mean_time_to_target=("time_to_target", "mean"),
        mean_anytime_auc=("anytime_auc", "mean"),
        n_runs=("score", "count"),
    )
    order = {c: i for i, c in enumerate(CONFIG_ORDER)}
    summary["_ord"] = summary["config"].map(order).fillna(999)
    summary = summary.sort_values(["budget", "_ord", "config"]).drop(columns=["_ord"])
    summary.to_csv(out / "summary_by_config.csv", index=False)
    print("\n=== summary by config ===")
    print(summary.round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PrepBench matrix")
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23, 29, 41])
    ap.add_argument("--budget", type=float, required=True, help="wall-clock seconds per run")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--configs", nargs="+", default=list(RUN_MATRIX.keys()), choices=list(RUN_MATRIX.keys()))
    ap.add_argument("--datasets", nargs="+", default=None, help="dataset names; default uses all Deepline datasets")
    ap.add_argument("--max-datasets", type=int, default=None, help="quick debug cap")
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--per-eval-timeout", type=float, default=None,
                    help="cap one evaluation's wall-clock seconds; default=budget/4")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--population-size", type=int, default=10)
    ap.add_argument("--offspring-size", type=int, default=10)
    ap.add_argument("--mutation-probability", type=float, default=0.9)
    ap.add_argument("--crossover-probability", type=float, default=0.7)
    ap.add_argument("--fitness-cost-penalty", type=float, default=1e-4,
                    help="EA survival penalty on estimated cost; use 0 for zero-penalty sensitivity")
    ap.add_argument("--fitness-complexity-penalty", type=float, default=1e-3,
                    help="EA survival penalty on graph complexity; use 0 for zero-penalty sensitivity")
    ap.add_argument("--run-tag", default="main",
                    help="label written to run_manifest.json for sensitivity runs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    timeout = args.per_eval_timeout if args.per_eval_timeout is not None else min(args.budget / 4, 60)
    base = SearchConfig(
        wall_clock_budget_seconds=args.budget,
        cv_folds=args.cv_folds,
        per_eval_timeout_seconds=timeout,
        population_size=args.population_size,
        offspring_size=args.offspring_size,
        mutation_probability=args.mutation_probability,
        crossover_probability=args.crossover_probability,
        fitness_cost_penalty=args.fitness_cost_penalty,
        fitness_complexity_penalty=args.fitness_complexity_penalty,
    )

    datasets = list(args.datasets) if args.datasets else sorted(ALL_DATASET_NAMES)
    if args.max_datasets:
        datasets = datasets[: args.max_datasets]

    completed = load_completed_runs(out / "experiment_records.csv")
    tasks = []
    for ds_name in datasets:
        for cfg_id in args.configs:
            for seed in args.seeds:
                key = (ds_name, cfg_id, int(seed), float(args.budget))
                if key not in completed:
                    tasks.append((ds_name, cfg_id, int(seed), float(args.budget), base, out))

    manifest = {
        "budget": args.budget,
        "seeds": args.seeds,
        "configs": args.configs,
        "datasets": datasets,
        "per_eval_timeout": timeout,
        "cv_folds": args.cv_folds,
        "population_size": args.population_size,
        "offspring_size": args.offspring_size,
        "mutation_probability": args.mutation_probability,
        "crossover_probability": args.crossover_probability,
        "fitness_cost_penalty": args.fitness_cost_penalty,
        "fitness_complexity_penalty": args.fitness_complexity_penalty,
        "run_tag": args.run_tag,
        "expected_runs": len(datasets) * len(args.configs) * len(args.seeds),
        "already_completed": len(completed),
        "pending_runs": len(tasks),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

    if tasks:
        lock = mp.Lock()
        with NoDaemonPool(processes=args.processes, initializer=init_worker, initargs=(lock,)) as pool:
            pool.starmap(process_one_setting, tasks)
    else:
        print("All requested runs are already complete.", flush=True)

    write_summaries(out)
    print(f"\nWrote CSVs to {out.resolve()}")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()

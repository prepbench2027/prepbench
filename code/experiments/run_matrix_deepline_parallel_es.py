#!/usr/bin/env python3
"""
Run the single-substrate benchmark RUN MATRIX
over datasets x seeds at one wall-clock budget, and write per-run + summary CSVs.

Examples
--------
# 1) Offline smoke run (no internet, tiny budget) -- just proves it all works:
python examples/run_matrix.py --smoke

# 2) Real run on OpenML datasets, 300s budget, 5 seeds:
python examples/run_matrix.py --datasets credit-g vehicle adult --seeds 0 1 2 3 4 --budget 300 --out results_300s

The RUN_MATRIX config IDs (L-S-F-A ... D-C-T-B) match the run-matrix table, so the numbers in summary_by_config.csv drop straight into the
paper's result tables.
"""
from __future__ import annotations

import argparse
import sys
import warnings
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# make the package importable when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv()

from parallel_evolutionary_pipeline.config import SearchConfig
from parallel_evolutionary_pipeline.datasets import OpenMLDataset, load_openml_dataset
from parallel_evolutionary_pipeline.search_earlystop import RUN_MATRIX, ParallelAwareEA
from data_loader.load_deepline import load_deepline_dataset, ALL_DATASET_NAMES

warnings.filterwarnings("ignore")
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# 全局锁，供各个子进程共享以保证写文件时的线程/进程安全
file_lock = None

def init_worker(lock):
    """进程池初始化函数：将主进程传过来的锁赋给全局变量"""
    global file_lock
    file_lock = lock


def _builtin_dataset(name: str) -> OpenMLDataset:
    """Offline fallback datasets (all-numeric) so --smoke runs without a network."""
    from sklearn import datasets as skd

    loaders = {
        "breast_cancer": skd.load_breast_cancer,
        "wine": skd.load_wine,
        "iris": skd.load_iris,
    }
    raw = loaders[name](as_frame=True)
    X = raw.data.copy()
    num = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    y = pd.Series(raw.target, name="target").astype("category")
    return OpenMLDataset(name=name, X=X.reset_index(drop=True), y=y.reset_index(drop=True),
                         numeric_columns=num, categorical_columns=cat)


def get_dataset(name: str, cache: str | None, offline: bool) -> OpenMLDataset:
    if offline:
        return _builtin_dataset(name)
    return load_openml_dataset(name, data_home=cache)


def _time_to_target(hist: pd.DataFrame, target: float) -> float:
    hit = hist.loc[hist["best_so_far"] >= target, "elapsed_seconds"]
    return float(hit.iloc[0]) if len(hit) else float("nan")


def _anytime_auc(hist: pd.DataFrame, budget: float, denom: float) -> float:
    if hist.empty or denom <= 0:
        return float("nan")
    h = hist.sort_values("elapsed_seconds")
    t = h["elapsed_seconds"].clip(upper=budget).to_numpy(dtype=float)
    b = h["best_so_far"].to_numpy(dtype=float)
    t = np.append(t, budget)
    b = np.append(b, b[-1])
    return float(_trapz(b, t) / budget / denom)


def load_completed_runs(rec_file: Path) -> set[tuple[str, str, int]]:
    """读取已有的 experiment_records.csv，返回已完成 (dataset, config, seed) 集合。"""
    if not rec_file.exists() or rec_file.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(rec_file, on_bad_lines="skip")
        required = {"dataset", "config", "seed"}
        if not required.issubset(df.columns):
            return set()
        completed = set()
        for _, row in df.iterrows():
            completed.add((str(row["dataset"]), str(row["config"]), int(row["seed"])))
        return completed
    except Exception:
        return set()


def _load_or_skip(ds_name: str) -> Any:
    """Load dataset or return None on failure."""
    try:
        ds = load_deepline_dataset(ds_name)
        print(f"[dataset] {ds_name}: n={ds.n_samples} d={ds.n_features} "
              f"(num={len(ds.numeric_columns)} cat={len(ds.categorical_columns)})")
        return ds
    except Exception as exc:
        print(f"[warn] could not load '{ds_name}' ({type(exc).__name__}); skipping")
        return None


def process_one_setting(ds_name: str, cfg_id: str, seed: int, budget: float,
                        base: SearchConfig, out_dir: Path) -> tuple[dict | None, list[dict]]:
    """
    处理单个实验 setting: one dataset × one config × one seed.
    每次得到结果后立刻以追加(append)模式安全地写入CSV中。
    返回 (record, run_history) 供主进程汇总。
    """
    dataset = _load_or_skip(ds_name)
    if dataset is None:
        return None, []

    conf = replace(base, random_state=seed, wall_clock_budget_seconds=budget)
    method = RUN_MATRIX[cfg_id]
    result = ParallelAwareEA(conf, method).fit(dataset)
    best = result.best_individual
    rh = pd.DataFrame(result.history)
    n_full = int((rh["status"] != "screened_out").sum()) if len(rh) else 0
    n_screen = int((rh["status"] == "screened_out").sum()) if len(rh) else 0

    # 单条记录
    record = dict(
        dataset=ds_name, config=cfg_id, seed=seed, budget=budget,
        score=best.full_score, estimated_cost=best.estimated_cost,
        serial_cost=best.serial_cost, runtime_seconds=best.runtime_seconds,
        branches=best.graph.branch_count(), nodes=best.graph.preprocessing_nodes(),
        n_full_evals=n_full, n_screened_out=n_screen, n_pruned=result.pruned,
    )

    # 历史记录列表
    run_history = []
    for row in result.history:
        run_history.append({"dataset": ds_name, "config": cfg_id, "seed": seed,
                            "budget": budget, **row})

    score = best.full_score if best.full_score is not None else float("nan")
    print(f"    [{ds_name}] {cfg_id:10s} seed={seed}  score={score:.4f}  "
          f"evals={n_full:3d}  pruned={result.pruned:3d}  branches={best.graph.branch_count()}")

    rec_file = out_dir / "experiment_records.csv"
    hist_file = out_dir / "search_history.csv"
    rec_df = pd.DataFrame([record])
    hist_df = pd.DataFrame(run_history)

    with file_lock:
        # header=not file.exists() 确保如果文件不存在，第一次写入时会带上表头
        rec_df.to_csv(rec_file, mode='a', index=False, header=not rec_file.exists())
        if not hist_df.empty:
            hist_df.to_csv(hist_file, mode='a', index=False, header=not hist_file.exists())

    return record, run_history


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-substrate benchmark run matrix")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 114, 256, 512, 768])
    ap.add_argument("--budget", type=float, default=300.0, help="wall-clock seconds per run")
    ap.add_argument("--out", default="benchmark_results", help="output directory")
    ap.add_argument("--cache", default=None, help="OpenML data_home cache directory")
    ap.add_argument("--processes", type=int, default=None, help="Number of parallel processes (default: cpu_count)")
    ap.add_argument("--per-eval-timeout", type=float, default=None,
                    help="cap one evaluation's wall-clock seconds; default = budget/4 in real runs")
    args = ap.parse_args()

    budget = args.budget
    timeout = args.per_eval_timeout
    if timeout is None:
        timeout = budget / 4
    base = SearchConfig(wall_clock_budget_seconds=budget, cv_folds=5, per_eval_timeout_seconds=timeout)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 保留 experiment_records.csv 和 search_history.csv 不断累积，用于断点恢复。
    # run_metrics.csv 和 summary_by_config.csv 每次会完整覆盖重写。
    rec_file = out / "experiment_records.csv"
    hist_file = out / "search_history.csv"

    # 读取已有记录，避免重复执行
    completed = load_completed_runs(rec_file)
    print(f"Found {len(completed)} already-completed runs in {rec_file}")

    # 构建任务列表 —— 每个任务是一个独立的 (dataset, config, seed) 组合
    tasks = []
    for ds_name in ALL_DATASET_NAMES:
        for cfg_id in RUN_MATRIX.keys():
            for seed in args.seeds:
                if (ds_name, cfg_id, seed) not in completed:
                    tasks.append((ds_name, cfg_id, seed, budget, base, out))

    n_tasks = len(tasks)
    if n_tasks == 0:
        print("All experiment settings already completed — nothing to run.")
    else:
        print(f"Starting parallel evaluation of {n_tasks} experiment settings "
              f"({len(ALL_DATASET_NAMES)} datasets × {len(RUN_MATRIX)} configs × {len(args.seeds)} seeds, "
              f"{len(completed)} already done) using Pool...")

        # 建立多进程安全锁
        lock = mp.Lock()

        # 启动进程池，并将锁传递到各工作进程
        with mp.Pool(processes=args.processes, initializer=init_worker, initargs=(lock,)) as pool:
            results = pool.starmap(process_one_setting, tasks)

        # 收集本次新增记录供日志打印（最终的 summary 将从文件完整读取）
        new_records: list[dict] = []
        new_histories: list[dict] = []
        for rec, hist in results:
            if rec is not None:
                new_records.append(rec)
            new_histories.extend(hist)

        print(f"\nCompleted {len(new_records)} new experiment settings.")

    # === 从文件读取全部记录（历史 + 本次新增）进行汇总计算 ===
    rec = pd.read_csv(rec_file, on_bad_lines="skip") if rec_file.exists() and rec_file.stat().st_size > 0 else pd.DataFrame()
    hist = pd.read_csv(hist_file, on_bad_lines="skip") if hist_file.exists() and hist_file.stat().st_size > 0 else pd.DataFrame()

    if not rec.empty:
        targets = (0.95 * rec.groupby("dataset")["score"].max()).to_dict()
        best_obs = rec.groupby("dataset")["score"].max().to_dict()
        rows = []
        for r in rec.itertuples(index=False):
            h = hist[(hist.dataset == r.dataset) & (hist.config == r.config) & (hist.seed == r.seed)] if not hist.empty else pd.DataFrame()
            rows.append(dict(
                dataset=r.dataset, config=r.config, seed=r.seed, score=r.score,
                n_full_evals=r.n_full_evals, n_pruned=r.n_pruned, branches=r.branches,
                time_to_target=_time_to_target(h, targets.get(r.dataset, float("nan"))) if len(h) else float("nan"),
                anytime_auc=_anytime_auc(h, r.budget, best_obs.get(r.dataset, 0.0)) if len(h) else float("nan"),
            ))
        met = pd.DataFrame(rows)
        # 覆盖写入 run_metrics.csv（而不是追加）
        met.to_csv(out / "run_metrics.csv", index=False)

        summary = met.groupby("config").agg(
            mean_score=("score", "mean"),
            mean_evals=("n_full_evals", "mean"),
            mean_pruned=("n_pruned", "mean"),
            mean_branches=("branches", "mean"),
            mean_time_to_target=("time_to_target", "mean"),
            mean_anytime_auc=("anytime_auc", "mean"),
        ).reindex([k for k in RUN_MATRIX.keys() if k in met["config"].unique()])
        summary.to_csv(out / "summary_by_config.csv")
        print("\n=== summary by config (mean over datasets x seeds) ===")
        print(summary.round(4).to_string())

    print(f"\nWrote CSVs to {out.resolve()}/")


if __name__ == "__main__":
    main()
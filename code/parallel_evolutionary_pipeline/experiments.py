from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from .config import SearchConfig
from .datasets import OPENML_DATASET_NAMES, load_openml_dataset
from .search_earlystop import METHOD_PRESETS, ParallelAwareEA


@dataclass
class ExperimentRecord:
    dataset: str
    method: str
    seed: int
    score: float | None
    estimated_cost: float | None
    serial_cost: float | None
    runtime_seconds: float | None
    branches: int
    nodes: int


@dataclass
class DatasetRecord:
    dataset: str
    n_samples: int
    n_features: int
    n_numeric: int
    n_categorical: int
    missing_fraction: float
    n_classes: int


def _config_with_seed(base: SearchConfig, seed: int) -> SearchConfig:
    cfg = SearchConfig(**asdict(base))
    cfg.random_state = seed
    return cfg


def run_openml_benchmark(
    output_dir: str | Path,
    dataset_names: list[str] | None = None,
    seeds: list[int] | None = None,
    base_config: SearchConfig | None = None,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = dataset_names or OPENML_DATASET_NAMES
    seeds = seeds or list(range(30))
    base_config = base_config or SearchConfig()
    records: list[ExperimentRecord] = []
    histories: list[dict] = []
    dataset_rows: list[DatasetRecord] = []

    for dataset_name in datasets:
        dataset = load_openml_dataset(dataset_name, data_home=output / "openml_cache")
        missing_fraction = float(dataset.X.isna().mean().mean()) if not dataset.X.empty else 0.0
        dataset_rows.append(
            DatasetRecord(
                dataset=dataset_name,
                n_samples=dataset.n_samples,
                n_features=dataset.n_features,
                n_numeric=len(dataset.numeric_columns),
                n_categorical=len(dataset.categorical_columns),
                missing_fraction=missing_fraction,
                n_classes=int(dataset.y.nunique()),
            )
        )
        for method_name, method_cfg in METHOD_PRESETS.items():
            for seed in seeds:
                search = ParallelAwareEA(_config_with_seed(base_config, seed), method_cfg)
                result = search.fit(dataset)
                best = result.best_individual
                records.append(
                    ExperimentRecord(
                        dataset=dataset_name,
                        method=method_name,
                        seed=seed,
                        score=best.full_score,
                        estimated_cost=best.estimated_cost,
                        serial_cost=best.serial_cost,
                        runtime_seconds=best.runtime_seconds,
                        branches=best.graph.branch_count(),
                        nodes=best.graph.preprocessing_nodes(),
                    )
                )
                for row in result.history:
                    histories.append(
                        {
                            "dataset": dataset_name,
                            "method": method_name,
                            "seed": seed,
                            **row,
                        }
                    )

    frame = pd.DataFrame([asdict(r) for r in records])
    summary = (
        frame.groupby(["dataset", "method"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            std_score=("score", "std"),
            mean_runtime=("runtime_seconds", "mean"),
            mean_branches=("branches", "mean"),
            mean_nodes=("nodes", "mean"),
        )
    )
    pairwise_rows = []
    pivot = frame.pivot_table(index=["dataset", "seed"], columns="method", values="score")
    if "proposed" in pivot.columns:
        for method in [m for m in pivot.columns if m != "proposed"]:
            paired = pivot[["proposed", method]].dropna()
            if paired.empty:
                continue
            delta = paired["proposed"] - paired[method]
            p = None if delta.eq(0).all() else float(wilcoxon(delta).pvalue)
            pairwise_rows.append(
                {
                    "baseline": method,
                    "wins": int((delta > 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "losses": int((delta < 0).sum()),
                    "mean_delta": float(delta.mean()),
                    "p_value": p,
                }
            )
    friedman = {}
    rank_pivot = pivot.dropna()
    if rank_pivot.shape[1] > 2 and not rank_pivot.empty:
        stat, p = friedmanchisquare(*[rank_pivot[c] for c in rank_pivot.columns])
        friedman = {"statistic": float(stat), "p_value": float(p)}

    records_path = output / "experiment_records.csv"
    summary_path = output / "experiment_summary.csv"
    pairwise_path = output / "pairwise_tests.csv"
    stats_path = output / "global_stats.json"
    history_path = output / "search_history.csv"
    datasets_path = output / "dataset_characteristics.csv"
    frame.to_csv(records_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(pairwise_rows).to_csv(pairwise_path, index=False)
    pd.DataFrame(histories).to_csv(history_path, index=False)
    pd.DataFrame([asdict(r) for r in dataset_rows]).to_csv(datasets_path, index=False)
    stats_path.write_text(pd.Series(friedman).to_json(), encoding="utf-8")
    return {
        "records": records_path.resolve(),
        "summary": summary_path.resolve(),
        "pairwise": pairwise_path.resolve(),
        "history": history_path.resolve(),
        "datasets": datasets_path.resolve(),
        "stats": stats_path.resolve(),
    }

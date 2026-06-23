#!/usr/bin/env python3
"""提取两个基准汇总指标：

1. Flat-Rand —— 56 个数据集上 5 个种子的 mean balanced accuracy。
2. Per-dataset best —— 对每个数据集，在所有 9 个配置中选出
   (5-seed-mean) 最高的那个 score，再取平均（oracle 式的上界）。

对 300s 和 600s 两个预算各输出一个 CSV。
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent

for budget in (300, 600):
    df = pd.read_csv(HERE / f"results/eab{budget}/experiment_records.csv")

    # ---- 1) Flat-Rand ----
    fr = df[df["config"] == "Flat-Rand"]
    flat = (
        fr.groupby("dataset", as_index=False)["score"]
        .mean()
        .rename(columns={"score": "flatrand_score"})
        .sort_values("dataset")
        .reset_index(drop=True)
    )

    # ---- 2) Per-dataset best (oracle) ----
    per_ds = df.groupby(["dataset", "config"], as_index=False)["score"].mean()
    best = (
        per_ds.loc[per_ds.groupby("dataset")["score"].idxmax()]
        .rename(columns={"config": "best_config", "score": "best_score"})
        .sort_values("dataset")
        .reset_index(drop=True)
    )

    # ---- Merge & write ----
    out = flat.merge(best, on="dataset", how="left")
    out.to_csv(HERE / f"flatrand_scores_{budget}.csv", index=False)

    print(f"=== {budget}s ===")
    print(f"  Flat-Rand mean:      {out['flatrand_score'].mean():.6f}")
    print(f"  Per-dataset best:    {out['best_score'].mean():.6f}")
    print(f"  Datasets:            {len(out)}")
    print()

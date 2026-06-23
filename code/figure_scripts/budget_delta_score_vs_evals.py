#!/usr/bin/env python3
"""budget_delta_score_vs_evals.pdf

Paired budget deltas. For each (dataset, config, seed) we pair the run across
budgets and compute the change in score. Left: 60s -> 300s (sizable gains).
Right: 300s -> 600s (negligible, clustered near zero). Each point is one
configuration's mean delta; the violin/strip shows the per-run distribution.

Run:  python3 budget_delta_score_vs_evals.py
Reads: experiment_records60.csv, experiment_records300.csv, experiment_records600.csv
Writes: ../figures/budget_delta_score_vs_evals.pdf

TODO：要不要将其改成“boxplot”？
"""
import os, csv
from collections import defaultdict
import numpy as np
from _style import apply_style, CONFIG_ORDER, config_color
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
F = {b: os.path.join(HERE, '..', f'results/eab{b}', f"experiment_records.csv") for b in (60, 300, 600)}
OUT = os.path.join(HERE, "..", "figures", "budget_delta_score_vs_evals.pdf")


def load(path):
    d = {}
    for r in csv.DictReader(open(path)):
        key = (r["dataset"], r["config"], r["seed"])
        d[key] = float(r["score"])
    return d


def main():
    apply_style()
    s60, s300, s600 = load(F[60]), load(F[300]), load(F[600])

    # paired deltas per config
    delta_a = defaultdict(list)   # 60 -> 300
    delta_b = defaultdict(list)   # 300 -> 600
    for key in s300:
        ds, cfg, seed = key
        if key in s60:
            delta_a[cfg].append(s300[key] - s60[key])
        if key in s600:
            delta_b[cfg].append(s600[key] - s300[key])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    panels = [("60s $\\to$ 300s", delta_a), ("300s $\\to$ 600s", delta_b)]
    cfgs = [c for c in CONFIG_ORDER if c in delta_a]

    for ax, (title, delta) in zip(axes, panels):
        for i, cfg in enumerate(cfgs):
            vals = np.array(delta[cfg])
            # strip of per-run deltas
            jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.5
            ax.scatter(np.full_like(vals, i) + jitter, vals,
                       s=6, alpha=0.25, color=config_color(cfg), zorder=2,
                       edgecolor="none")
            # mean marker
            ax.scatter([i], [vals.mean()], s=80, color=config_color(cfg),
                       edgecolor="black", linewidth=0.7, zorder=4)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels(cfgs, rotation=45, ha="right", fontsize=7)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(-0.25, 1)

    axes[0].set_ylabel("$\\Delta$ balanced accuracy")
    # fig.suptitle("Paired budget deltas (each dot = one run; large = config mean)",
                #  fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)

    print("\nconfig         mean(60->300)  mean(300->600)")
    for cfg in cfgs:
        print(f"  {cfg:12s}  {np.mean(delta_a[cfg]):+.4f}        {np.mean(delta_b[cfg]):+.4f}")


if __name__ == "__main__":
    main()

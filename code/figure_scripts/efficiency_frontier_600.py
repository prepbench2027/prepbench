#!/usr/bin/env python3
"""efficiency_frontier_600.pdf

Accuracy vs. number of full evaluations per run at the 600s budget.
Each point is one configuration's (mean full-evals, mean score) over all
datasets/seeds. Random controls (stars) sit upper-left: higher accuracy with
fewer full evaluations. Structured/cost-aware configs trade evaluation count at
statistically equivalent accuracy.

Run:  python3 efficiency_frontier_600.py
Reads: experiment_records600.csv   (cols: config, score, n_full_evals, ...)
Writes: ../figures/efficiency_frontier_600.pdf
"""
import os, csv
from collections import defaultdict
import numpy as np
from _style import apply_style, CONFIG_ORDER, config_color, RANDOM_CONTROLS
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results/eab600/experiment_records.csv")
OUT = os.path.join(HERE, "..", "figures", "efficiency_frontier_600.pdf")


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["score"] = float(r["score"])
        r["n_full_evals"] = float(r["n_full_evals"])
    return rows

shift_config = {
    'D-C-T-B': (-10, 0.0003),
    'D-S-F-B': (0, -0.001),
    'D-S-F-A': (-200, 0),
    'D-C-F-A': (0, -0.0013),
}

def main():
    apply_style()
    rows = load(CSV)
    # mean score and mean full-evals per config (over all dataset/seed runs)
    score = defaultdict(list)
    fevals = defaultdict(list)
    for r in rows:
        score[r["config"]].append(r["score"])
        fevals[r["config"]].append(r["n_full_evals"])

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for cfg in CONFIG_ORDER:
        if cfg not in score:
            continue
        x = np.mean(fevals[cfg])
        y = np.mean(score[cfg])
        is_rand = cfg in RANDOM_CONTROLS
        ax.scatter(x, y,
                   marker="*" if is_rand else "o",
                   s=240 if is_rand else 70,
                   color=config_color(cfg),
                   edgecolor="black", linewidth=0.6,
                   zorder=3)
        # label with a small offset
        offset_x, offset_y = shift_config.get(cfg, (0, 0))
        ax.annotate(cfg, (x+offset_x, y+offset_y),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Mean full evaluations per run (log scale)")
    ax.set_ylabel("Mean balanced accuracy")
    ax.set_title("Efficiency frontier (600s)")

    # legend for the two marker classes
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#1b7837",
               markeredgecolor="black", markersize=14, label="Random controls"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#b2182b",
               markeredgecolor="black", markersize=8, label="Structured / cost-aware"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#6a51a3",
               markeredgecolor="black", markersize=8, label="Simple structure"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)

    # also print the underlying numbers for the caption/verification
    print("\nconfig            mean_fevals   mean_score")
    for cfg in CONFIG_ORDER:
        if cfg in score:
            print(f"  {cfg:12s}   {np.mean(fevals[cfg]):10.1f}   {np.mean(score[cfg]):.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""fallback_by_config_60.pdf

Fallback runs at the 60s budget by configuration. A fallback run has a positive
score but zero completed full evaluations (score>0 and n_full_evals==0): the
reported score came from a fallback path, not a completed candidate evaluation.
Random controls never fall back; structured/staged configs do, because their
preparation plans more often fail to finish a full evaluation in time.

Run:  python3 fallback_by_config_60.py
Reads: experiment_records60.csv
Writes: ../figures/fallback_by_config_60.pdf
"""
import os, csv
from collections import defaultdict
import numpy as np
from _style import apply_style, CONFIG_ORDER, config_color
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results/eab60/experiment_records.csv")
OUT = os.path.join(HERE, "..", "figures", "fallback_by_config_60.pdf")


def main():
    apply_style()
    rows = list(csv.DictReader(open(CSV)))
    total = defaultdict(int)
    fallback = defaultdict(int)
    for r in rows:
        cfg = r["config"]
        total[cfg] += 1
        if float(r["score"]) > 0 and float(r["n_full_evals"]) == 0:
            fallback[cfg] += 1

    cfgs = [c for c in CONFIG_ORDER if c in total]
    counts = [fallback[c] for c in cfgs]
    pcts = [100.0 * fallback[c] / total[c] for c in cfgs]
    colors = [config_color(c) for c in cfgs]

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    bars = ax.bar(range(len(cfgs)), counts, color=colors,
                  edgecolor="black", linewidth=0.6)
    for i, (b, p) in enumerate(zip(bars, pcts)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{p:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels(cfgs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fallback runs (count)")
    ax.set_title("Fallback runs at 60s (score $>$ 0, zero full evals)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)

    print("\nconfig         fallback / total   (%)")
    for c in cfgs:
        print(f"  {c:12s}  {fallback[c]:4d} / {total[c]:4d}   {100*fallback[c]/total[c]:5.1f}%")


if __name__ == "__main__":
    main()

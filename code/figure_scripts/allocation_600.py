#!/usr/bin/env python3
"""allocation_600.pdf

Evaluation allocation at 600s. Shows how candidates are spent: completed full
evaluations vs. screened-out (staged) vs. pruned (budget-aware). D-C-T-B
performs fewer full evaluations than the full-evaluation DAG baseline (D-S-F-A)
and relies on screening and pruning -- but this allocation shift does not raise
final accuracy (see Wilcoxon table in the paper).

Run:  python3 allocation_600.py
Reads: experiment_records600.csv
Writes: ../figures/allocation_600.pdf
"""
import os, csv
from collections import defaultdict
import numpy as np
from _style import apply_style
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results/eab600/experiment_records.csv")
OUT = os.path.join(HERE, "..", "figures", "allocation_600.pdf")

# Configs to contrast: a full-eval DAG baseline vs the combined staged+pruned one.
SHOW = ["D-S-F-A", "D-C-T-B"]
LABELS = {"D-S-F-A": "D-S-F-A\n(full-eval DAG)", "D-C-T-B": "D-C-T-B\n(staged+pruned)"}


def main():
    apply_style()
    rows = list(csv.DictReader(open(CSV)))
    full = defaultdict(list)
    scr = defaultdict(list)
    pru = defaultdict(list)
    for r in rows:
        c = r["config"]
        full[c].append(float(r["n_full_evals"]))
        scr[c].append(float(r["n_screened_out"]))
        pru[c].append(float(r["n_pruned"]))

    means = {c: (np.mean(full[c]), np.mean(scr[c]), np.mean(pru[c])) for c in SHOW}

    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    x = np.arange(len(SHOW))
    f = [means[c][0] for c in SHOW]
    s = [means[c][1] for c in SHOW]
    p = [means[c][2] for c in SHOW]

    b1 = ax.bar(x, f, color="#2166ac", edgecolor="black", linewidth=0.6,
                label="Full evaluations")
    b2 = ax.bar(x, s, bottom=f, color="#f4a582", edgecolor="black",
                linewidth=0.6, label="Screened out (staged)")
    b3 = ax.bar(x, p, bottom=np.array(f) + np.array(s), color="#b2182b",
                edgecolor="black", linewidth=0.6, label="Pruned (budget)")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in SHOW], fontsize=8)
    ax.set_ylabel("Mean candidates per run")
    ax.set_title("Evaluation allocation (600s)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=True, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)

    print("\nconfig     full_evals  screened_out  pruned")
    for c in SHOW:
        print(f"  {c:8s}  {means[c][0]:10.1f}  {means[c][1]:12.1f}  {means[c][2]:8.1f}")


if __name__ == "__main__":
    main()

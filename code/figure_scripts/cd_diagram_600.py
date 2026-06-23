#!/usr/bin/env python3
"""cd_diagram_600.pdf

Critical-difference (CD) diagram (Nemenyi post-hoc, alpha=0.05) at the 600s
budget. Lower average rank is better. Configurations connected by a horizontal
bar are NOT significantly different. The random controls sit at the best (low)
ranks, separated from the structured/cost-aware cluster.

Ranking protocol: for each dataset we average score over seeds, rank the 9
configs (1 = best), then average ranks across datasets (Friedman ranking).
The CD is computed manually from the studentized-range critical value so the
script has no external dependency beyond numpy/scipy.

Run:  python3 cd_diagram_600.py
Reads: experiment_records600.csv
Writes: ../figures/cd_diagram_600.pdf

"""
import os, csv
from collections import defaultdict
import numpy as np
from _style import apply_style, config_color, RANDOM_CONTROLS
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results/eab600/experiment_records.csv")
OUT = os.path.join(HERE, "..", "figures", "cd_diagram_600.pdf")

# Studentized range critical values q_alpha for alpha=0.05, infinite df,
# indexed by k (number of groups). Standard Nemenyi/Demsar table.
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
       8: 3.031, 9: 3.102, 10: 3.164}


def main():
    apply_style()
    rows = list(csv.DictReader(open(CSV)))
    # The paper excludes home_credit (incomplete across budgets); match that.
    for r in rows:
        r["score"] = float(r["score"])

    # dataset -> config -> mean score over seeds
    ds_cfg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ds_cfg[r["dataset"]][r["config"]].append(r["score"])
    configs = sorted({r["config"] for r in rows})
    k = len(configs)

    # rank per dataset (1 = best), average across datasets
    rank_sum = defaultdict(float)
    n_ds = 0
    for ds, cfgs in ds_cfg.items():
        if set(cfgs) != set(configs):
            continue
        n_ds += 1
        means = {c: np.mean(cfgs[c]) for c in configs}
        # higher score = better = lower rank
        ordered = sorted(configs, key=lambda c: -means[c])
        # handle ties by average rank
        scores = np.array([means[c] for c in ordered])
        ranks = np.empty(len(ordered))
        i = 0
        while i < len(ordered):
            j = i
            while j + 1 < len(ordered) and scores[j + 1] == scores[i]:
                j += 1
            avg = np.mean(range(i + 1, j + 2))
            for t in range(i, j + 1):
                ranks[t] = avg
            i = j + 1
        for c, rk in zip(ordered, ranks):
            rank_sum[c] += rk
    avg_rank = {c: rank_sum[c] / n_ds for c in configs}

    # Critical difference
    q = Q05[k]
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n_ds))

    # order configs by rank (best first)
    order = sorted(configs, key=lambda c: avg_rank[c])
    ranks_sorted = [avg_rank[c] for c in order]

    # ---- draw the CD diagram ----
    lo = int(np.floor(min(avg_rank.values())))
    hi = int(np.ceil(max(avg_rank.values())))
    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(0.23, 1)
    ax.axis("off")

    # top axis
    yaxis = 0.85
    ax.plot([lo, hi], [yaxis, yaxis], color="black", linewidth=1.0)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [yaxis, yaxis + 0.03], color="black", linewidth=1.0)
        ax.text(t, yaxis + 0.06, str(t), ha="center", va="bottom", fontsize=8)
    ax.text((lo + hi) / 2, yaxis + 0.13, "Average rank (lower is better)",
            ha="center", fontsize=8)

    # place labels: half on the left, half on the right
    half = (len(order) + 1) // 2
    left = order[:half]
    right = order[half:][::-1]

    def draw_label(cfg, rank, side, row):
        y = yaxis - 0.12 - row * 0.11
        elbow_x = (lo - 0.2) if side == "left" else (hi + 0.2)
        ax.plot([rank, rank], [yaxis, y], color=config_color(cfg), linewidth=1.2)
        ax.plot([rank, elbow_x], [y, y], color=config_color(cfg), linewidth=1.2)
        ha = "right" if side == "left" else "left"
        tx = elbow_x - 0.05 if side == "left" else elbow_x + 0.05
        weight = "bold" if cfg in RANDOM_CONTROLS else "normal"
        ax.text(tx, y, f"{cfg} ({rank:.2f})", ha=ha, va="center",
                fontsize=8, color=config_color(cfg), fontweight=weight)

    for i, cfg in enumerate(left):
        draw_label(cfg, avg_rank[cfg], "left", i)
    for i, cfg in enumerate(right):
        draw_label(cfg, avg_rank[cfg], "right", i)

    # CD bar (length = cd) shown as a reference near the top-left
    bar_y = yaxis + 0.20
    ax.plot([lo, lo + cd], [bar_y, bar_y], color="black", linewidth=2.5)
    ax.text(lo + cd / 2, bar_y - 0.05, f"CD = {cd:.3f}", ha="center",
            va="bottom", fontsize=8)

    # cliques: connect groups whose rank span <= CD
    clique_y = yaxis - 0.04
    used = []
    sorted_ranks = sorted(avg_rank.values())
    i = 0
    level = 0
    bars = []
    for a_i in range(len(order)):
        a = avg_rank[order[a_i]]
        # find furthest config within CD
        group = [order[j] for j in range(len(order)) if 0 <= avg_rank[order[j]] - a <= cd]
        if len(group) > 1:
            r0 = a
            r1 = max(avg_rank[g] for g in group)
            bars.append((r0, r1))
    # dedup nested bars
    bars = sorted(set(bars))
    drawn = []
    for (r0, r1) in bars:
        if any(r0 >= d0 and r1 <= d1 for (d0, d1) in drawn):
            continue
        drawn.append((r0, r1))
    for idx, (r0, r1) in enumerate(drawn):
        yy = clique_y - idx * 0.035
        ax.plot([r0 - 0.05, r1 + 0.05], [yy, yy], color="gray", linewidth=3.0,
                solid_capstyle="round")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)

    print(f"\nn_datasets={n_ds}, k={k}, q05={q}, CD={cd:.4f}")
    print("config         avg_rank")
    for c in order:
        print(f"  {c:12s}  {avg_rank[c]:.3f}")


if __name__ == "__main__":
    main()

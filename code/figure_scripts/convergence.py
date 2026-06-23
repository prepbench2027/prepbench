#!/usr/bin/env python3
"""convergence_600.pdf

Convergence curves: mean best-so-far balanced accuracy as a function of
wall-clock time at the 600s budget.  Shows how quickly each configuration's
incumbent improves as evaluations accumulate, with ±1 SE ribbons.

Run:  python3 convergence.py
Reads: results/eab600/search_history.csv
Writes: ../figures/convergence_600.pdf
"""
import os, sys
import numpy as np
import pandas as pd
from _style import apply_style, CONFIG_ORDER, config_color, RANDOM_CONTROLS
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE_PATH = os.path.join(HERE, "..", "results/eab600/search_history.csv")
OUT = os.path.join(HERE, "..", "figures", "convergence_600.pdf")

# Configs to show — all except the purely linear/fixed-branch full-eval ones
# so the plot stays readable.  Keep at least one random control and the staged
# + cost-aware methods.
KEEP_CONFIGS = [
    "D-C-T-B",
    "D-S-T-A",
    "D-C-F-A",
    "D-S-F-B",
    "D-S-F-A",
    "Flat-Rand",
    "D-Rand",
]

BUDGET_SEC = 600.0
N_GRID = 21                     # 0 %, 5 %, …, 100 %
GRID = np.linspace(0, 1, N_GRID)


def main():
    apply_style()

    # ---------- read ----------
    usecols = ["dataset", "config", "seed", "budget",
               "elapsed_seconds", "status", "best_so_far"]
    df = pd.read_csv(TRACE_PATH, usecols=usecols,
                     dtype={"budget": float, "seed": int},
                     low_memory=False)

    # Filter to the right budget and configs
    df = df[df["budget"] == BUDGET_SEC]
    df = df[df["config"].isin(KEEP_CONFIGS)]

    # Only rows with a completed evaluation (status == "ok") that moved the
    # incumbent — but we also keep rows that just confirmed the existing best.
    df = df[df["status"].astype(str).str.strip().str.lower() == "ok"]
    df = df.dropna(subset=["elapsed_seconds", "best_so_far"])
    df["elapsed_seconds"] = pd.to_numeric(df["elapsed_seconds"], errors="coerce")
    df["best_so_far"] = pd.to_numeric(df["best_so_far"], errors="coerce")
    df = df.dropna(subset=["elapsed_seconds", "best_so_far"])

    print(f"  {len(df):,} ok-eval rows across {KEEP_CONFIGS} configs", flush=True)

    # ---------- interpolate per (dataset, config, seed) ----------
    rows = []
    for (ds, cfg, seed), g in df.groupby(["dataset", "config", "seed"],
                                         sort=False):
        g = g.sort_values("elapsed_seconds")
        t = np.clip(g["elapsed_seconds"].to_numpy() / BUDGET_SEC, 0, 1)
        bsf = g["best_so_far"].to_numpy()

        # Deduplicate time points
        ut, ui = np.unique(t, return_index=True)
        ub = bsf[ui]

        # Interpolate onto the grid; left/right fill for before/after extremes
        interp = np.interp(GRID, ut, ub, left=ub[0], right=ub[-1])
        for frac, val in zip(GRID, interp):
            rows.append({
                "config": cfg,
                "time_frac": frac,
                "best_so_far": val,
            })

    curve = pd.DataFrame(rows)

    # ---------- aggregate ----------
    agg = (
        curve
        .groupby(["config", "time_frac"], as_index=False)
        .agg(
            mean=("best_so_far", "mean"),
            se=("best_so_far", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            n=("best_so_far", "count"),
        )
    )

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(4.6, 3.2))

    for cfg in KEEP_CONFIGS:
        sub = agg[agg["config"] == cfg]
        if sub.empty:
            continue
        t = sub["time_frac"].values * BUDGET_SEC
        y = sub["mean"].values
        se = sub["se"].values
        color = config_color(cfg)
        ls = "--" if cfg in RANDOM_CONTROLS else "-"
        lw = 1.2 if cfg in RANDOM_CONTROLS else 1.8
        ax.plot(t, y, color=color, ls=ls, lw=lw, label=cfg, zorder=3)
        ax.fill_between(t, y - se, y + se, color=color, alpha=0.15, lw=0)

    ax.set_xlim(0, BUDGET_SEC)
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Mean best-so-far balanced accuracy")
    ax.set_title("Convergence at 600s budget")
    ax.legend(frameon=True, fontsize=7, loc="lower right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

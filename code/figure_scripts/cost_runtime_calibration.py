#!/usr/bin/env python3
"""cost_runtime_calibration.pdf

Static cost estimate vs. realized per-evaluation runtime at 600s. Each point is
one run: x = estimated_cost (the static estimate that drives pruning), y =
realized runtime per full evaluation (runtime_seconds / n_full_evals). A weak
association means the estimate used for pruning is only loosely coupled to the
runtime that actually consumes the budget. The R^2 is computed from the data
and printed; update the caption to match.

Run:  python3 cost_runtime_calibration.py
Reads: experiment_records600.csv
Writes: ../figures/cost_runtime_calibration.pdf
"""
import os, csv
import numpy as np
from _style import apply_style
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results/eab60/experiment_records.csv")
OUT = os.path.join(HERE, "..", "figures", "cost_runtime_calibration60.pdf")


def main():
    apply_style()
    rows = list(csv.DictReader(open(CSV)))

    xs, ys = [], []
    for r in rows:
        est = float(r["estimated_cost"])
        rt = float(r["runtime_seconds"])
        if est > 0 and rt > 0:
            xs.append(est)
            ys.append(rt)          # total realized runtime the estimate predicts
    xs = np.array(xs)
    ys = np.array(ys)

    # Linear R^2 (the quantity reported in the paper): how well the static cost
    # estimate predicts the realized runtime it is meant to model.
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
    pred = slope * xs + intercept
    ss_res = np.sum((ys - pred) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.scatter(xs, ys, s=10, alpha=0.3, color="#2166ac", edgecolor="none")
    xx = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(xx, slope * xx + intercept, color="#b2182b", linewidth=1.5,
            label=f"linear fit ($R^2={r2:.2f}$)")
    ax.set_xlabel("Static cost estimate (estimated_cost)")
    ax.set_ylabel("Realized runtime (s)")
    ax.set_title("Cost estimate vs. realized runtime (600s)")
    ax.legend(frameon=True, loc="upper left")
    ax.set_ylim(-40, 700)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)
    print(f"\nn={len(xs)} runs, linear R^2 = {r2:.4f}")
    print(">>> Caption fig:cost should state this R^2.")


if __name__ == "__main__":
    main()

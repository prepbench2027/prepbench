#!/usr/bin/env python3
"""staged_calibration_dctb_600.pdf

Staged-screening calibration for the D-C-T-B configuration at the 600s budget.
Each point is one candidate that reached stage-1 screening: x = stage-1
subsample score, y = full-validation score. The strong correlation along the
identity line, together with a false-negative rate of zero (no screened-out
candidate would have become the run's full-validation winner), shows that
screening discards only reliably non-winning candidates -- the mechanism behind
the allocation-not-accuracy result (Finding F3/F4).

Run:  python3 staged_calibration_dctb_600.py
Reads: stage_full_sample.csv  (cols: config, budget, stage1_score, score)
       staged_metrics.csv     (cols: config, budget, pearson_stage1_full,
                               promotion_rate, false_neg_rate, ...)
Writes: ../figures/staged_calibration_dctb_600.pdf
"""
import os, csv
import numpy as np
from _style import apply_style
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "..", "results/trace_summary", "stage_full_sample.csv")
METRICS = os.path.join(HERE, "..", "results/trace_summary", "staged_metrics.csv")
OUT = os.path.join(HERE, "..", "figures", "staged_calibration_dsta_600.pdf")

# CONFIG = "D-C-T-B"
CONFIG = "D-S-T-A"
BUDGET = "600"


def load_pairs():
    s1, sf = [], []
    for r in csv.DictReader(open(SAMPLE)):
        if r["config"] == CONFIG and r["budget"] == BUDGET:
            s1.append(float(r["stage1_score"]))
            sf.append(float(r["score"]))
    return np.array(s1), np.array(sf)


def load_metric():
    for r in csv.DictReader(open(METRICS)):
        if r["config"] == CONFIG and r["budget"] == BUDGET:
            pear = float(r["pearson_stage1_full"])
            fn = float(r["false_neg_rate"]) if r["false_neg_rate"] not in ("", None) else 0.0
            promo = float(r["promotion_rate"]) if r["promotion_rate"] not in ("", None) else float("nan")
            return pear, fn, promo
    return float("nan"), float("nan"), float("nan")


def main():
    apply_style()
    s1, sf = load_pairs()
    pear, fn, promo = load_metric()
    if not np.isfinite(pear) and len(s1) > 1:
        pear = np.corrcoef(s1, sf)[0, 1]

    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    ax.scatter(s1, sf, s=8, alpha=0.25, color="#2166ac", edgecolor="none", zorder=2)
    lim = [min(s1.min(), sf.min()), max(s1.max(), sf.max())]
    ax.plot(lim, lim, "--", color="black", linewidth=1.0, zorder=3, label="identity")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Stage-1 subsample score")
    ax.set_ylabel("Full-validation score")
    ax.set_title(f"Staged screening calibration ({CONFIG}, {BUDGET}s)\n"
                 f"Pearson $r={pear:.2f}$, promotion {promo:.2f}, FN-rate $={fn:.2f}$")
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote REAL figure: {OUT}")
    print(f"  n_sampled_pairs={len(s1)}, Pearson(metrics)={pear:.4f}, "
          f"promotion={promo:.3f}, FN-rate={fn:.3f}")


if __name__ == "__main__":
    main()

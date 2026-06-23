import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _style import apply_style, config_color


# =========================
# 1. Config
# =========================

HERE = os.path.dirname(os.path.abspath(__file__))

TRACE_PATH = os.path.join(HERE, "..", "results/eab600/search_history.csv")

OUT_SUMMARY = os.path.join(HERE, "..", "figures/convergence_summary_600.csv")
OUT_FIG_PDF = os.path.join(HERE, "..", "figures/fig_convergence_600.pdf")
OUT_FIG_PNG = os.path.join(HERE, "..", "figures/fig_convergence_600.png")

BUDGET = 600

# 主文推荐只画这 4 条，清楚
# KEEP_CONFIGS = [
#     "D-S-F-A",      # plain DAG baseline
#     "D-C-T-B",      # combined cost-aware/staged/pruned
#     "D-Rand",       # random feasible DAG
#     "Flat-Rand",    # flat random
# ]

# 如果你想画 7 条，用这个替换上面的 KEEP_CONFIGS
KEEP_CONFIGS = [
    "D-S-F-A",
    "D-C-F-A",
    "D-S-F-B",
    "D-S-T-A",
    "D-C-T-B",
    # "D-Rand",
    # "Flat-Rand",
]

GRID = np.linspace(0, BUDGET, 25)  # 0, 25, 50, ..., 600


# =========================
# 2. Column mapping
# =========================
# 如果你的 trace 字段名不一样，在这里改

COLUMN_ALIASES = {
    "dataset": ["dataset", "dataset_id", "task", "task_id", "openml_task_id"],
    "seed": ["seed", "random_seed"],
    "config": ["config", "configuration", "method", "config_id"],
    "budget": ["budget", "time_budget", "wallclock_budget"],
    "elapsed_time": ["elapsed_time", "elapsed", "time", "wall_time", "elapsed_sec", "elapsed_seconds"],
    "status": ["status", "candidate_status"],
    "full_score": ["full_score", "score", "balanced_accuracy", "full_validation_score"],
    "score_source": ["score_source", "source"],
    "candidate_index": ["candidate_index", "candidate_idx", "idx", "index"],
}


def resolve_columns(path):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    resolved = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in header:
                resolved[canonical] = a
                break

    required = ["dataset", "seed", "config", "elapsed_time", "full_score"]
    missing = [c for c in required if c not in resolved]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns are:\n{header}\n"
            f"Please update COLUMN_ALIASES."
        )

    return resolved, header


# =========================
# 3. Load and filter trace
# =========================

def load_filtered_trace(path):
    resolved, header = resolve_columns(path)

    usecols = list(set(resolved.values()))
    chunks = []

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000):
        # Rename to canonical names
        rename_map = {v: k for k, v in resolved.items()}
        chunk = chunk.rename(columns=rename_map)

        # Budget filter if budget column exists
        if "budget" in chunk.columns:
            chunk["budget"] = pd.to_numeric(chunk["budget"], errors="coerce")
            chunk = chunk[chunk["budget"] == BUDGET]

        # Config filter
        chunk = chunk[chunk["config"].isin(KEEP_CONFIGS)]

        # Status filter if status exists
        if "status" in chunk.columns:
            status = chunk["status"].astype(str).str.lower()
            ok_status = [
                "ok",
                "completed",
                "complete",
                "success",
                "succeeded",
                "full",
                "full_candidate",
            ]
            chunk = chunk[status.isin(ok_status)]

        # Score source filter if score_source exists
        if "score_source" in chunk.columns:
            source = chunk["score_source"].astype(str).str.lower()
            ok_sources = [
                "full_candidate",
                "full",
                "full_validation",
                "candidate_full",
            ]
            chunk = chunk[source.isin(ok_sources)]

        chunk["elapsed_time"] = pd.to_numeric(chunk["elapsed_time"], errors="coerce")
        chunk["full_score"] = pd.to_numeric(chunk["full_score"], errors="coerce")
        chunk["candidate_index"] = pd.to_numeric(chunk["candidate_index"], errors="coerce")

        # For configs that lack elapsed_seconds (D-Rand, Flat-Rand — no
        # evolutionary loop timing), estimate time from candidate_index.
        # Assumes random-search candidates are roughly evenly distributed
        # through the wall-clock budget.
        if "candidate_index" in chunk.columns:
            mask_missing_time = chunk["elapsed_time"].isna()
            if mask_missing_time.any():
                max_idx = chunk["candidate_index"].max()
                if max_idx > 0:
                    chunk.loc[mask_missing_time, "elapsed_time"] = (
                        chunk.loc[mask_missing_time, "candidate_index"] / max_idx * BUDGET
                    )

        chunk = chunk.dropna(subset=["elapsed_time", "full_score"])

        # Keep valid range
        chunk = chunk[(chunk["elapsed_time"] >= 0) & (chunk["elapsed_time"] <= BUDGET)]

        if len(chunk) > 0:
            chunks.append(chunk)

    if not chunks:
        raise ValueError("No rows left after filtering. Check status/score_source/config/budget filters.")

    df = pd.concat(chunks, ignore_index=True)

    print("Filtered rows:", len(df))
    print("Configs found:", sorted(df["config"].unique()))
    print("Datasets:", df["dataset"].nunique())
    print("Seeds:", sorted(df["seed"].unique())[:20])

    return df


# =========================
# 4. Build convergence summary
# =========================

def make_convergence_summary(df):
    rows = []

    group_cols = ["dataset", "seed", "config"]

    for (dataset, seed, config), g in df.groupby(group_cols):
        g = g.sort_values("elapsed_time")

        t = g["elapsed_time"].to_numpy(dtype=float)
        s = g["full_score"].to_numpy(dtype=float)

        if len(t) == 0:
            continue

        best = np.maximum.accumulate(s)

        # For duplicated times, keep the maximum best-so-far at that time
        tmp = pd.DataFrame({"time": t, "best": best})
        tmp = tmp.groupby("time", as_index=False)["best"].max()
        t_unique = tmp["time"].to_numpy()
        best_unique = tmp["best"].to_numpy()

        # Ensure the curve starts at time 0.
        # Before first completed full evaluation, use first observed incumbent.
        if t_unique[0] > 0:
            t_unique = np.insert(t_unique, 0, 0.0)
            best_unique = np.insert(best_unique, 0, best_unique[0])

        # Ensure the curve reaches budget end.
        if t_unique[-1] < BUDGET:
            t_unique = np.append(t_unique, BUDGET)
            best_unique = np.append(best_unique, best_unique[-1])

        interp_best = np.interp(GRID, t_unique, best_unique)

        for time_sec, val in zip(GRID, interp_best):
            rows.append({
                "dataset": dataset,
                "seed": seed,
                "config": config,
                "time_sec": time_sec,
                "time_frac": time_sec / BUDGET,
                "best_so_far": val,
            })

    curve = pd.DataFrame(rows)

    summary = (
        curve
        .groupby(["config", "time_sec", "time_frac"], as_index=False)
        .agg(
            mean_best_so_far=("best_so_far", "mean"),
            std_best_so_far=("best_so_far", "std"),
            n_runs=("best_so_far", "count"),
        )
    )

    summary["se_best_so_far"] = summary["std_best_so_far"] / np.sqrt(summary["n_runs"])
    summary["ci95_low"] = summary["mean_best_so_far"] - 1.96 * summary["se_best_so_far"]
    summary["ci95_high"] = summary["mean_best_so_far"] + 1.96 * summary["se_best_so_far"]

    return curve, summary


# =========================
# 5. Plot
# =========================

def plot_convergence(summary):
    apply_style()
    plt.figure(figsize=(2.8, 2.3))

    # 线型区分，避免全靠颜色
    line_styles = {
        "D-S-F-A": "-",
        "D-C-T-B": "--",
        "D-Rand": "-.",
        "Flat-Rand": ":",
        "D-C-F-A": "-",
        "D-S-F-B": "--",
        "D-S-T-A": "-.",
    }

    # 显示顺序
    plot_order = [c for c in KEEP_CONFIGS if c in summary["config"].unique()]

    for config in plot_order:
        g = summary[summary["config"] == config].sort_values("time_sec")

        x = g["time_sec"].to_numpy()
        y = g["mean_best_so_far"].to_numpy()
        lo = g["ci95_low"].to_numpy()
        hi = g["ci95_high"].to_numpy()

        plt.plot(
            x,
            y,
            label=config,
            # color=config_color(config),
            linewidth=1.2,
            linestyle=line_styles.get(config, "-"),
        )

        # 置信带可以保留；如果太乱，可以注释掉这段
        # plt.fill_between(x, lo, hi, alpha=0.12)

    plt.xlabel("Wall-clock time (s)")
    plt.ylabel("Mean best-so-far balanced accuracy")
    # plt.title("Incumbent convergence under the 600s budget")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_FIG_PDF, bbox_inches="tight")
    plt.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to {OUT_FIG_PDF}")
    print(f"Saved figure to {OUT_FIG_PNG}")


# =========================
# 6. Main
# =========================

if __name__ == "__main__":
    df = load_filtered_trace(TRACE_PATH)
    curve, summary = make_convergence_summary(df)

    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"Saved summary to {OUT_SUMMARY}")

    print("\nFinal values at 600s:")
    final = summary[summary["time_sec"] == BUDGET].sort_values(
        "mean_best_so_far",
        ascending=False
    )
    print(final[["config", "mean_best_so_far", "ci95_low", "ci95_high", "n_runs"]])

    plot_convergence(summary)
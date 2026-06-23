import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

CTXPIPE = {
    "Accident_Casualties": 0.2674,
    "Frogs_MFCCs_family": 0.9637,
    "Frogs_MFCCs_gen": 0.9523,
    "Frogs_MFCCs_spec": 0.9551,
    "HTRU_2": 0.8783,
    "IndianLiverPatientDatasetILPD": 0.5449,
    "Iris": 1.0000,
    "LendingClubIssuedLoans": 0.0876,
    "Skin_NonSkin": 0.9987,
    "The_broken_machine": 0.4887,
    "Wine_classification": 1.0000,
    "adult": 0.7722,
    "analcatdata_broadwaymult": 0.4472,
    "analcatdata_germangss": 0.0304,
    "ar4": 0.6429,
    "bank-full": 0.6939,
    "baseball": 0.7245,
    "biodeg": 0.8429,
    "blood-transfusion": 0.6309,
    "bodyfat": 0.9828,
    "braziltourism": 0.1938,
    "bureau": 0.5013,
    "car": 0.8502,
    "chatfield_4": 0.8130,
    "cmc": 0.4874,
    "crx": 0.8154,
    "data_banknote_authentication": 1.0000,
    "dermatology": 0.9514,
    "diggle_table_a2": 1.0000,
    "disclosure_z": 0.5620,
    "glass": 0.6150,
    "haberman": 0.5865,
    "home_credit": 0.5121,
    "imagesegmentation": 0.9682,
    "kc3": 0.6111,
    "kidney": 0.6500,
    "magic04": 0.8299,
    "mammographic_masses": 0.8048,
    "movement_libras": 0.8113,
    "no2": 0.5341,
    "plasma_retinol": 0.5891,
    "pm10": 0.5855,
    "schizo": 0.5473,
    "socmob": 0.8356,
    "solar-flare": 0.6867,
    "test_bng_cmc": 0.5588,
    "test_breast": 0.9642,
    "test_credit": 0.6690,
    "test_eucalyptus": 0.4931,
    "test_ilpd": 0.6154,
    "test_irish": 0.9911,
    "test_phoneme": 0.7717,
    "triazines": 0.7632,
    "veteran": 0.7515,
    "weatherAUS": 0.7797,
    "wilt": 0.8492,
}

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Cannot find any of columns: {candidates}. Existing columns: {list(df.columns)}")

def load_run_metrics(run_dir: Path, budget: int):
    path = run_dir / "run_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    dataset_col = find_col(df, ["dataset", "dataset_name", "task", "task_name"])
    config_col = find_col(df, ["config", "method", "configuration"])
    seed_col = find_col(df, ["seed", "random_state"])
    score_col = find_col(df, ["score", "balanced_accuracy", "test_score", "best_score"])
    df = df.rename(columns={
        dataset_col: "dataset",
        config_col: "config",
        seed_col: "seed",
        score_col: "score",
    })
    if "budget" not in df.columns:
        df["budget"] = budget
    return df

def coverage_audit(df):
    rows = []
    for budget, g0 in df.groupby("budget"):
        datasets = sorted(g0["dataset"].unique())
        configs = sorted(g0["config"].unique())
        seeds = sorted(g0["seed"].unique())
        expected = len(datasets) * len(configs) * len(seeds)
        observed = g0[["dataset", "config", "seed"]].drop_duplicates().shape[0]
        rows.append({
            "budget": budget,
            "datasets": len(datasets),
            "configs": len(configs),
            "seeds": len(seeds),
            "expected_runs": expected,
            "observed_runs": observed,
            "missing_runs": expected - observed,
        })
    return pd.DataFrame(rows)

def summary_by_config(df):
    return (
        df.groupby(["budget", "config"])
        .agg(
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            std_score=("score", "std"),
            n_runs=("score", "size"),
            n_datasets=("dataset", "nunique"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
        .sort_values(["budget", "mean_score"], ascending=[True, False])
    )

def dataset_config_scores(df):
    return (
        df.groupby(["budget", "dataset", "config"])
        .agg(score=("score", "mean"))
        .reset_index()
    )

def friedman_and_wilcoxon(df):
    rows = []
    dcs = dataset_config_scores(df)
    for budget, g in dcs.groupby("budget"):
        pivot = g.pivot(index="dataset", columns="config", values="score")
        pivot = pivot.dropna()
        configs = list(pivot.columns)
        if len(configs) >= 3:
            stat, p = friedmanchisquare(*[pivot[c].values for c in configs])
            rows.append({
                "budget": budget,
                "test": "friedman",
                "comparison": "all_configs",
                "statistic": stat,
                "p_value": p,
                "n_datasets": pivot.shape[0],
            })
        for base in ["D-Rand", "Flat-Rand"]:
            if base not in pivot.columns:
                continue
            for c in configs:
                if c == base:
                    continue
                diff = pivot[c] - pivot[base]
                try:
                    wstat, wp = wilcoxon(diff.values, zero_method="wilcox")
                except ValueError:
                    wstat, wp = np.nan, np.nan
                rows.append({
                    "budget": budget,
                    "test": "wilcoxon_signed_rank",
                    "comparison": f"{c} minus {base}",
                    "statistic": wstat,
                    "p_value": wp,
                    "mean_diff": diff.mean(),
                    "median_diff": diff.median(),
                    "wins": int((diff > 0).sum()),
                    "ties": int((diff == 0).sum()),
                    "losses": int((diff < 0).sum()),
                    "n_datasets": pivot.shape[0],
                })
    return pd.DataFrame(rows)

def tost_sensitivity(df, margins=(0.005, 0.01, 0.02)):
    # 这里先输出 paired mean diff 和是否落在 margin 内；
    # 严格 TOST p-value 如果你原代码已有，可以之后替换。
    rows = []
    dcs = dataset_config_scores(df)
    for budget, g in dcs.groupby("budget"):
        pivot = g.pivot(index="dataset", columns="config", values="score").dropna()
        for base in ["D-Rand", "Flat-Rand"]:
            if base not in pivot.columns:
                continue
            for c in pivot.columns:
                if c == base:
                    continue
                diff = pivot[c] - pivot[base]
                for m in margins:
                    rows.append({
                        "budget": budget,
                        "comparison": f"{c} minus {base}",
                        "margin": m,
                        "mean_diff": diff.mean(),
                        "median_diff": diff.median(),
                        "within_margin_mean": abs(diff.mean()) <= m,
                        "within_margin_median": abs(diff.median()) <= m,
                        "n_datasets": pivot.shape[0],
                    })
    return pd.DataFrame(rows)

def ctxpipe_comparison(df):
    ctx = pd.DataFrame({"dataset": list(CTXPIPE.keys()), "ctxpipe_score": list(CTXPIPE.values())})
    dcs = dataset_config_scores(df)
    rows = []
    for budget, g in dcs.groupby("budget"):
        pivot = g.pivot(index="dataset", columns="config", values="score").reset_index()
        merged = ctx.merge(pivot, on="dataset", how="inner")
        for method in ["D-Rand", "Flat-Rand"]:
            if method not in merged.columns:
                continue
            diff = merged["ctxpipe_score"] - merged[method]
            rows.append({
                "budget": budget,
                "comparison": f"CtxPipe minus {method}",
                "ctxpipe_mean": merged["ctxpipe_score"].mean(),
                "method_mean": merged[method].mean(),
                "mean_diff": diff.mean(),
                "median_diff": diff.median(),
                "ctxpipe_wins": int((diff > 0).sum()),
                "ties": int((diff == 0).sum()),
                "ctxpipe_losses": int((diff < 0).sum()),
                "n_datasets": merged.shape[0],
            })
        score_cols = [c for c in merged.columns if c not in ["dataset", "ctxpipe_score"]]
        if score_cols:
            merged["best_controlled"] = merged[score_cols].max(axis=1)
            diff = merged["ctxpipe_score"] - merged["best_controlled"]
            rows.append({
                "budget": budget,
                "comparison": "CtxPipe minus best_controlled",
                "ctxpipe_mean": merged["ctxpipe_score"].mean(),
                "method_mean": merged["best_controlled"].mean(),
                "mean_diff": diff.mean(),
                "median_diff": diff.median(),
                "ctxpipe_wins": int((diff > 0).sum()),
                "ties": int((diff == 0).sum()),
                "ctxpipe_losses": int((diff < 0).sum()),
                "n_datasets": merged.shape[0],
            })
    return pd.DataFrame(rows)

def trace_audit(run_dirs, budgets):
    rows = []
    for run_dir, budget in zip(run_dirs, budgets):
        run_path = run_dir / "run_metrics.csv"
        trace_path = run_dir / "search_history.csv"
        if not run_path.exists() or not trace_path.exists():
            continue
        run = pd.read_csv(run_path)
        trace = pd.read_csv(trace_path)
        for col in ["dataset", "config", "seed"]:
            if col not in run.columns or col not in trace.columns:
                continue
        run_keys = run[["dataset", "config", "seed"]].drop_duplicates()
        trace_keys = trace[["dataset", "config", "seed"]].drop_duplicates()
        merged = run_keys.merge(trace_keys, on=["dataset", "config", "seed"], how="left", indicator=True)
        rows.append({
            "budget": budget,
            "run_metrics_runs": len(run_keys),
            "trace_runs": len(trace_keys),
            "missing_trace_runs": int((merged["_merge"] == "left_only").sum()),
            "trace_rows": len(trace),
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run300", required=True)
    ap.add_argument("--run600", required=True)
    ap.add_argument("--out", default="paper_result_bundle")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    run_dirs = [Path(args.run300), Path(args.run600)]
    budgets = [300, 600]
    df = pd.concat([load_run_metrics(d, b) for d, b in zip(run_dirs, budgets)], ignore_index=True)

    df.to_csv(out / "dataset_config_seed_scores.csv", index=False)
    coverage_audit(df).to_csv(out / "coverage_audit.csv", index=False)
    summary_by_config(df).to_csv(out / "summary_by_config.csv", index=False)
    dataset_config_scores(df).to_csv(out / "dataset_config_scores.csv", index=False)
    friedman_and_wilcoxon(df).to_csv(out / "stat_tests.csv", index=False)
    tost_sensitivity(df).to_csv(out / "tost_sensitivity.csv", index=False)
    ctxpipe_comparison(df).to_csv(out / "ctxpipe_paired_comparison.csv", index=False)
    trace_audit(run_dirs, budgets).to_csv(out / "trace_audit.csv", index=False)

    ctx = pd.DataFrame({"dataset": list(CTXPIPE.keys()), "ctxpipe_score": list(CTXPIPE.values())})
    ctx.to_csv(out / "ctxpipe_scores.csv", index=False)

    print(f"Written result bundle to: {out.resolve()}")
    print("Files:")
    for p in sorted(out.glob("*.csv")):
        print(" -", p.name, p.stat().st_size, "bytes")

if __name__ == "__main__":
    main()
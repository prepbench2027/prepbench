#!/usr/bin/env python3
"""
Contextual external baselines for the VLDB EAB revision.

These baselines are NOT attribution baselines for the factorized run matrix.
They are contextual reference points showing how the benchmark substrate relates
to common default ML pipelines and optional AutoML systems under the same dataset
list, seeds, metric, and wall-clock budgets.

Built-in systems that require only scikit-learn:
  sklearn_logreg_default
  sklearn_rf_default
  sklearn_hgb_default

Optional systems, run only if installed:
  flaml              pip install "flaml[automl]"
  autogluon          pip install autogluon.tabular
  autosklearn        pip install auto-sklearn
  tpot               pip install tpot

Example
-------
python external_baselines/run_contextual_baselines.py \
  --datasets Accident_Casualties Frogs_MFCCs_family \
  --systems sklearn_logreg_default,sklearn_rf_default,sklearn_hgb_default,flaml \
  --budgets 300 600 \
  --seeds 7 11 23 29 41 \
  --out external_contextual_300_600 \
  --processes 8
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from data_loader.load_deepline import ALL_DATASET_NAMES, load_deepline_Xy

warnings.filterwarnings("ignore")

BUILTINS = {"sklearn_logreg_default", "sklearn_rf_default", "sklearn_hgb_default"}
OPTIONAL_IMPORTS = {
    "flaml": "flaml",
    "autogluon": "autogluon.tabular",
    "autosklearn": "autosklearn.classification",
    "tpot": "tpot",
}


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    num = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return num, cat


def make_preprocessor(X: pd.DataFrame, *, scale_numeric: bool = True, encode: str = "onehot") -> ColumnTransformer:
    num, cat = split_columns(X)
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("sc", StandardScaler()))
    if encode == "ordinal":
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        try:
            cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # scikit-learn < 1.2
            cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", cat_encoder)]), cat),
    ], remainder="drop")


def run_builtin(system: str, Xtr: pd.DataFrame, ytr: pd.Series, Xte: pd.DataFrame, seed: int) -> np.ndarray:
    if system == "sklearn_logreg_default":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1)
        pipe = Pipeline([("pre", make_preprocessor(Xtr, scale_numeric=True, encode="onehot")), ("clf", clf)])
    elif system == "sklearn_rf_default":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=1)
        pipe = Pipeline([("pre", make_preprocessor(Xtr, scale_numeric=False, encode="ordinal")), ("clf", clf)])
    elif system == "sklearn_hgb_default":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(random_state=seed, class_weight="balanced")
        pipe = Pipeline([("pre", make_preprocessor(Xtr, scale_numeric=False, encode="ordinal")), ("clf", clf)])
    else:
        raise ValueError(system)
    pipe.fit(Xtr, ytr)
    return pipe.predict(Xte)


def run_flaml(Xtr, ytr, Xte, budget: int, seed: int):
    from flaml import AutoML
    def balacc_metric(X_val, y_val, estimator, labels, X_train, y_train, weight_val=None, weight_train=None, *args):
        return 1.0 - balanced_accuracy_score(y_val, estimator.predict(X_val)), {}
    automl = AutoML()
    automl.fit(Xtr, ytr, task="classification", metric=balacc_metric, time_budget=int(budget), seed=int(seed), verbose=0)
    return automl.predict(Xte)


def run_autogluon(Xtr, ytr, Xte, budget: int, seed: int):
    from autogluon.tabular import TabularPredictor
    import tempfile
    train = Xtr.copy()
    label = "__target__"
    train[label] = ytr.reset_index(drop=True).astype(str)
    with tempfile.TemporaryDirectory() as tmp:
        predictor = TabularPredictor(label=label, eval_metric="balanced_accuracy", path=tmp, verbosity=0)
        predictor.fit(train, time_limit=int(budget), presets="medium_quality", num_bag_folds=0, num_stack_levels=0)
        return predictor.predict(Xte.copy())


def run_autosklearn(Xtr, ytr, Xte, budget: int, seed: int):
    import autosklearn.classification as askl
    from autosklearn.metrics import balanced_accuracy as askl_balacc
    cls = askl.AutoSklearnClassifier(
        time_left_for_this_task=int(budget),
        per_run_time_limit=max(30, int(budget) // 6),
        metric=askl_balacc,
        seed=int(seed),
        memory_limit=8192,
        n_jobs=1,
    )
    cls.fit(Xtr.copy(), ytr.copy())
    return cls.predict(Xte.copy())


def run_tpot(Xtr, ytr, Xte, budget: int, seed: int):
    from tpot import TPOTClassifier
    generations = max(2, min(10, int(budget) // 120))
    clf = TPOTClassifier(
        generations=generations,
        population_size=20,
        max_time_mins=max(1, int(budget) // 60),
        scoring="balanced_accuracy",
        random_state=int(seed),
        verbosity=0,
        n_jobs=1,
        config_dict="TPOT light",
    )
    # TPOT expects numeric arrays; use a standard preprocessing pipeline first.
    pre = make_preprocessor(Xtr, scale_numeric=True, encode="onehot")
    Xt = pre.fit_transform(Xtr, ytr)
    Xv = pre.transform(Xte)
    clf.fit(Xt, ytr)
    return clf.predict(Xv)


OPTIONAL_RUNNERS = {
    "flaml": run_flaml,
    "autogluon": run_autogluon,
    "autosklearn": run_autosklearn,
    "tpot": run_tpot,
}


def available_systems(requested: list[str]) -> list[str]:
    ok = []
    for s in requested:
        if s in BUILTINS:
            ok.append(s)
            continue
        module = OPTIONAL_IMPORTS.get(s)
        if module is None:
            print(f"[warn] unknown system '{s}', skipping", flush=True)
            continue
        try:
            __import__(module)
            ok.append(s)
        except Exception as exc:
            print(f"[warn] optional system '{s}' unavailable: {exc}; skipping", flush=True)
    return ok


def run_one(args):
    dataset, system, seed, budget, test_size = args
    try:
        X, y, _, _ = load_deepline_Xy(dataset)
        # Stratification is important for balanced accuracy and class-imbalanced datasets.
        stratify = y if pd.Series(y).nunique(dropna=False) > 1 else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=int(seed), stratify=stratify
        )
        t0 = time.perf_counter()
        if system in BUILTINS:
            yhat = run_builtin(system, Xtr, ytr, Xte, int(seed))
            effective_budget = 0
        else:
            yhat = OPTIONAL_RUNNERS[system](Xtr, ytr, Xte, int(budget), int(seed))
            effective_budget = int(budget)
        runtime = time.perf_counter() - t0
        score = balanced_accuracy_score(yte, yhat)
        return {
            "dataset": dataset,
            "config": system,
            "seed": int(seed),
            "budget": int(effective_budget),
            "requested_budget": int(budget),
            "score": float(score),
            "runtime_seconds": float(runtime),
            "status": "ok",
        }, None
    except Exception as exc:
        return None, {
            "dataset": dataset,
            "config": system,
            "seed": int(seed),
            "budget": int(budget),
            "error": repr(exc),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None, help="dataset names; default uses all Deepline datasets")
    ap.add_argument("--systems", default="sklearn_logreg_default,sklearn_rf_default,sklearn_hgb_default,flaml")
    ap.add_argument("--budgets", nargs="+", type=int, default=[300, 600])
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23, 29, 41])
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--max-datasets", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    systems = available_systems([s.strip() for s in args.systems.split(",") if s.strip()])
    if not systems:
        raise SystemExit("No requested systems are available.")
    datasets = sorted(args.datasets) if args.datasets else sorted(ALL_DATASET_NAMES)
    if args.max_datasets:
        datasets = datasets[: args.max_datasets]

    results_path = out / "external_contextual_results.csv"
    failures_path = out / "external_contextual_failures.jsonl"
    fields = ["dataset", "config", "seed", "budget", "requested_budget", "score", "runtime_seconds", "status"]

    done = set()
    if results_path.exists() and results_path.stat().st_size > 0:
        prev = pd.read_csv(results_path, on_bad_lines="skip")
        for r in prev.itertuples(index=False):
            done.add((str(r.dataset), str(r.config), int(r.seed), int(r.requested_budget)))
        print(f"[resume] {len(done)} completed external runs found", flush=True)

    work = []
    for ds in datasets:
        for system in systems:
            budgets = args.budgets if system not in BUILTINS else [args.budgets[0]]
            for budget in budgets:
                for seed in args.seeds:
                    key = (ds, system, int(seed), int(budget))
                    if key not in done:
                        work.append((ds, system, int(seed), int(budget), float(args.test_size)))

    manifest = {
        "datasets": datasets,
        "systems": systems,
        "budgets": args.budgets,
        "seeds": args.seeds,
        "test_size": args.test_size,
        "pending": len(work),
    }
    (out / "external_contextual_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

    fh = open(results_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields)
    if fh.tell() == 0:
        writer.writeheader()
        fh.flush()
    fail_fh = open(failures_path, "a", encoding="utf-8")

    rows = []
    if args.processes and args.processes > 1:
        with Pool(args.processes) as pool:
            iterator = pool.imap_unordered(run_one, work)
            for rec, fail in iterator:
                if rec:
                    writer.writerow(rec); fh.flush(); rows.append(rec)
                    print(f"{rec['dataset']:28s} {rec['config']:24s} seed={rec['seed']} budget={rec['requested_budget']} balacc={rec['score']:.4f}", flush=True)
                else:
                    fail_fh.write(json.dumps(fail) + "\n"); fail_fh.flush()
                    print(f"[fail] {fail}", flush=True)
    else:
        for item in work:
            rec, fail = run_one(item)
            if rec:
                writer.writerow(rec); fh.flush(); rows.append(rec)
                print(f"{rec['dataset']:28s} {rec['config']:24s} seed={rec['seed']} budget={rec['requested_budget']} balacc={rec['score']:.4f}", flush=True)
            else:
                fail_fh.write(json.dumps(fail) + "\n"); fail_fh.flush()
                print(f"[fail] {fail}", flush=True)
    fh.close(); fail_fh.close()

    res = pd.read_csv(results_path, on_bad_lines="skip") if results_path.exists() and results_path.stat().st_size > 0 else pd.DataFrame(rows)
    if not res.empty:
        summ = res.groupby(["config", "requested_budget"], as_index=False).agg(
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            std_score=("score", "std"),
            n_runs=("score", "count"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        summ.to_csv(out / "external_contextual_summary.csv", index=False)
        print("\n=== external contextual summary ===")
        print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

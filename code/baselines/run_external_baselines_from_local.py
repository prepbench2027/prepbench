#!/usr/bin/env python3
"""
run_external_baselines.py  --  produce reference results from real external
systems (FLAML, Auto-sklearn) and cheap default-model baselines on the SAME
datasets/seeds/budgets as the PrepBench-EAB benchmark, in a schema that joins to
experiment_records (columns: dataset, config, seed, budget, score).

Why: EAB reviewers expect a benchmark to evaluate existing, published
systems of wide interest. This script adds those reference points so the
factorized configurations can be compared against real AutoML systems.

PROTOCOL (documented, and meant to be matched to your benchmark): for each
(dataset, seed) we make a seeded stratified train/test split; each system is fit
on the training split under the wall-clock budget and scored by balanced
accuracy on the held-out test split. For a strict head-to-head, evaluate your
own configurations under the same split (see --note in the README we print).

SYSTEMS (run whichever are installed / requested):
  flaml            -- FLAML AutoML            (pip install "flaml[automl]")
  autosklearn      -- Auto-sklearn            (pip install auto-sklearn; Linux)
  rf_default       -- single RandomForest     (always available; budget-free)
  logreg_default   -- single LogisticRegression(always available; budget-free)

USAGE
  # 1) generate a manifest template from your run records, then fill it in:
  python run_external_baselines.py --records . --make-manifest manifest.csv
  # 2) run (defaults: 300s budget, 3 seeds, cheap baselines + flaml):
  python run_external_baselines.py --manifest manifest.csv --out external_out \
         --systems rf_default,logreg_default,flaml --budgets 300 --seeds 42 114 256
Upload the small external_out/ folder when done.
"""
import sys
from pathlib import Path

# Load .env BEFORE importing numpy/sklearn so that OMP_NUM_THREADS, etc. are
# honoured when those libraries initialise their thread pools.
from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
    print(f"[dotenv] loaded {_env_path}", flush=True)
else:
    load_dotenv()  # fallback: try CWD
    print(f"[dotenv] no .env at {_env_path}, tried CWD", flush=True)

import argparse
import json
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader.load_deepline import load_deepline_Xy

warnings.filterwarnings("ignore")
EXCLUDE = "home_credit"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def make_manifest(records_dir, path):
    names = set()
    for b in (60, 300, 600):
        fp = os.path.join(records_dir, f"experiment_records{b}.csv")
        if os.path.exists(fp):
            names |= set(pd.read_csv(fp, usecols=["dataset"]).dataset.unique())
    names = sorted(n for n in names if n != EXCLUDE)
    if not names:
        print("WARNING: found no experiment_records{60,300,600}.csv under "
              f"'{records_dir}', so the manifest is EMPTY. Point --records at the "
              "folder containing those files, or create the manifest by hand with "
              "columns: dataset, openml_task_id, openml_data_id, csv_path, target.")
    pd.DataFrame({"dataset": names, "openml_task_id": "", "openml_data_id": "",
                  "csv_path": "", "target": ""}).to_csv(path, index=False)
    print(f"Wrote manifest template with {len(names)} datasets to {path}.\n"
          "Fill in EITHER openml_task_id, OR openml_data_id + target, OR "
          "csv_path + target for each row, then re-run with --manifest.")


def build_openml_lookup():
    """Return lookup(name) -> (data_id, target, [candidate_ids]) or None."""
    import openml
    allds = openml.datasets.list_datasets(output_format="dataframe")

    def lookup(name):
        cand = allds[allds["name"] == name]
        if len(cand) == 0:
            cand = allds[allds["name"].str.lower() == name.lower()]
        if len(cand) == 0:
            return None
        c = cand.copy()
        if "status" in c.columns:
            act = c[c["status"] == "active"]
            if len(act):
                c = act
        sort_cols = [x for x in ("version", "did") if x in c.columns]
        if sort_cols:
            c = c.sort_values(sort_cols)
        did = int(c.iloc[0]["did"])
        try:
            ds = openml.datasets.get_dataset(did, download_data=False)
            target = ds.default_target_attribute or ""
        except Exception:
            target = ""
        return did, target, sorted(int(x) for x in cand["did"].tolist())

    return lookup


def split_columns(X):
    from pandas.api.types import is_numeric_dtype
    num = [c for c in X.columns if is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return num, cat


def preprocess_pipeline(X, estimator):
    num, cat = split_columns(X)
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ], remainder="drop")
    return Pipeline([("pre", pre), ("clf", estimator)])


# --------------------------------------------------------------------------- #
# System runners: each returns predictions on X_test
# --------------------------------------------------------------------------- #
def run_default(name, Xtr, ytr, Xte, seed):
    if name == "rf_default":
        from sklearn.ensemble import RandomForestClassifier
        est = RandomForestClassifier(n_estimators=300, random_state=seed,
                                     class_weight="balanced", n_jobs=1)
    else:  # logreg_default
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipe = preprocess_pipeline(Xtr, est)
    pipe.fit(Xtr, ytr)
    return pipe.predict(Xte)


def run_flaml(Xtr, ytr, Xte, budget, seed):
    from flaml import AutoML
    def balacc(X_val, y_val, estimator, labels, X_train, y_train,
               weight_val=None, weight_train=None, *args):
        s = balanced_accuracy_score(y_val, estimator.predict(X_val))
        return 1 - s, {"balacc": s}
    automl = AutoML()
    try:
        automl.fit(Xtr, ytr, task="classification", metric=balacc,
                   time_budget=budget, seed=seed, verbose=0)
    except Exception:
        automl.fit(Xtr, ytr, task="classification", metric="macro_f1",
                   time_budget=budget, seed=seed, verbose=0)
    return automl.predict(Xte)


def run_autosklearn(Xtr, ytr, Xte, budget, seed):
    import autosklearn.classification as askl
    from autosklearn.metrics import balanced_accuracy as askl_balacc
    cls = askl.AutoSklearnClassifier(
        time_left_for_this_task=budget,
        per_run_time_limit=max(30, budget // 6),
        metric=askl_balacc, seed=seed, memory_limit=8192)
    cls.fit(Xtr.copy(), ytr.copy())
    return cls.predict(Xte)


RUNNERS = {"flaml": run_flaml, "autosklearn": run_autosklearn}
DEFAULTS = {"rf_default", "logreg_default"}
_INSTALL = {"flaml": 'pip install "flaml[automl]"',
            "autosklearn": "pip install auto-sklearn  (Linux; also needs swig)"}


def check_systems(systems):
    """Keep only systems that are available; warn once about missing ones."""
    ok = []
    for s in systems:
        if s in DEFAULTS:
            ok.append(s)
            continue
        if s not in RUNNERS:
            print(f"WARNING: unknown system '{s}', skipping.", flush=True)
            continue
        try:
            __import__(s)
            ok.append(s)
        except Exception as e:
            print(f"WARNING: '{s}' is not installed ({e}); skipping it.\n"
                  f"         to enable it: {_INSTALL[s]}", flush=True)
    return ok


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _worker_run_combination(args):
    """Worker for multiprocessing: run one (dataset, seed, system, budget) combo.

    *args* is a tuple: (dataset, seed, system, budget, test_size).
    Returns (record_dict, failure_dict) — exactly one is *None*.
    """
    dataset, seed, system, budget, test_size = args
    try:
        X, y, _, _ = load_deepline_Xy(dataset)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=seed)
        t0 = time.time()
        if system in DEFAULTS:
            yhat = run_default(system, Xtr, ytr, Xte, seed)
        else:
            yhat = RUNNERS[system](Xtr, ytr, Xte, budget, seed)
        score = balanced_accuracy_score(yte, yhat)
        rec = dict(dataset=dataset, config=system, seed=seed,
                   budget=budget, score=round(float(score), 6),
                   runtime_seconds=round(time.time() - t0, 2))
        return rec, None
    except Exception as e:
        return None, dict(dataset=dataset, config=system, seed=seed,
                          budget=budget, error=str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=".")
    ap.add_argument("--make-manifest", default=None,
                    help="write a manifest template here and exit")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default="external_out")
    ap.add_argument("--systems", default="rf_default,logreg_default,flaml")
    ap.add_argument("--budgets", type=int, nargs="+", default=[300])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 114, 256])
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--max-datasets", type=int, default=None,
                    help="cap number of datasets (for a quick first pass)")
    ap.add_argument("--pool-size", type=int, default=32,
                    help="number of worker processes for parallel execution")
    args = ap.parse_args()

    if args.make_manifest:
        make_manifest(args.records, args.make_manifest)
        return
    if not args.manifest:
        raise SystemExit("Provide --manifest (or --make-manifest to create one).")

    os.makedirs(args.out, exist_ok=True)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if not systems:
        raise SystemExit("ERROR: --systems is empty.")
    systems = check_systems(systems)
    if not systems:
        raise SystemExit(
            "ERROR: none of the requested systems are available. Install one "
            '(e.g. pip install "flaml[automl]") or use the built-in baselines '
            "--systems rf_default,logreg_default.")
    man = pd.read_csv(args.manifest)
    if "dataset" not in man.columns:
        raise SystemExit(f"ERROR: manifest '{args.manifest}' has no 'dataset' "
                         f"column. Columns found: {list(man.columns)}")
    # man = man[man.dataset != EXCLUDE]
    if args.max_datasets:
        man = man.head(args.max_datasets)
    if len(man) == 0:
        raise SystemExit(
            f"ERROR: manifest '{args.manifest}' has no rows to process. Generate "
            "it with --make-manifest (with --records pointing at your "
            "experiment_records*.csv) and make sure it lists your datasets.")

    # incremental output + resume: append each result as it is computed
    results_path = os.path.join(args.out, "external_results.csv")
    fields = ["dataset", "config", "seed", "budget", "score", "runtime_seconds"]
    done, prior_rows = set(), []
    if os.path.exists(results_path) and os.path.getsize(results_path) > 0:
        try:
            prev = pd.read_csv(results_path, on_bad_lines="skip")
            if "dataset" not in prev.columns:
                raise ValueError("missing header")
            prior_rows = prev.dropna(subset=["dataset", "config", "seed",
                                             "budget"]).to_dict("records")
            for r in prior_rows:
                done.add((r["dataset"], r["config"], int(r["seed"]),
                          int(r["budget"])))
            print(f"resuming: {len(done)} results already on disk will be "
                  "skipped.", flush=True)
        except Exception as e:
            print(f"NOTE: existing {results_path} is empty/unreadable ({e}); "
                  "starting it fresh.", flush=True)
            os.remove(results_path)
            prior_rows = []
    import csv as _csv
    fh = open(results_path, "a", newline="")
    writer = _csv.DictWriter(fh, fieldnames=fields)
    if fh.tell() == 0:
        writer.writeheader()
        fh.flush()

    # Build flat work-item list for the multiprocessing pool.
    work_items = []
    for _, row in man.iterrows():
        ds = row["dataset"]
        for seed in args.seeds:
            for sysname in systems:
                budgets = (args.budgets if sysname in RUNNERS
                           else [args.budgets[0]])
                for budget in budgets:
                    if (ds, sysname, seed, budget) not in done:
                        work_items.append((ds, seed, sysname, budget,
                                           args.test_size))

    rows, failures = [], []
    if not work_items:
        print("No pending work items; all combinations already complete.",
              flush=True)
    else:
        print(f"Processing {len(work_items)} work item(s) with "
              f"{args.pool_size} worker(s)...", flush=True)
        with Pool(args.pool_size) as pool:
            for rec, failure in pool.imap_unordered(_worker_run_combination,
                                                    work_items):
                if rec is not None:
                    writer.writerow(rec)
                    fh.flush()
                    rows.append(rec)
                    done.add((rec["dataset"], rec["config"], rec["seed"],
                              rec["budget"]))
                    print(f"   {rec['dataset']:24} {rec['config']:14} "
                          f"seed={rec['seed']} b={rec['budget']} "
                          f"balacc={rec['score']:.4f}", flush=True)
                else:
                    failures.append(failure)
                    print(f"   [{failure['config']} FAIL] {failure['dataset']} "
                          f"seed={failure['seed']}: {failure['error']}",
                          flush=True)
    fh.close()

    all_rows = prior_rows + rows
    res = pd.DataFrame(all_rows)
    if len(res):
        summ = res.groupby(["config", "budget"]).score.agg(["mean", "std", "count"])
        summ.reset_index().to_csv(os.path.join(args.out, "external_summary.csv"),
                                  index=False)
        print("\n=== external summary (mean balanced accuracy) ===")
        print(summ.round(4).to_string())
    with open(os.path.join(args.out, "external_failures.json"), "w") as f:
        json.dump(failures, f, indent=2)
    print(f"\nResults at {results_path} ({len(all_rows)} rows total, "
          f"{len(rows)} new this run, {len(failures)} failures). Upload the folder.")


if __name__ == "__main__":
    main()

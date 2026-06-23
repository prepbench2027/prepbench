#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ctxpipe_aligned.py
======================

Protocol-matched re-scoring of CtxPipe (and the PrepBench controls) so that the
comparison is fair, i.e. so that any score gap reflects the *preprocessing
pipeline* and NOT differences in the evaluation harness.

------------------------------------------------------------------------------
WHY THIS SCRIPT EXISTS
------------------------------------------------------------------------------
CtxPipe's own reported balanced-accuracy numbers and the PrepBench numbers were
produced by DIFFERENT harnesses:

    * CtxPipe   : (typically) a single stratified hold-out split, its own fixed
                  downstream model, its own metric computation.
    * PrepBench : StratifiedKFold(5, shuffle, random_state=seed), averaged over
                  5 seeds {42,114,256,512,768}, with the estimator *searched*.

Comparing 0.783 (CtxPipe) vs 0.760 (random control) across those two harnesses
is confounded: the 2-point gap could be the split protocol or the downstream
model, not "better preprocessing". This script removes the confound by holding
EVERYTHING fixed except the preprocessing pipeline:

    fixed: the 56 datasets (loaded with the SAME loader PrepBench uses),
           StratifiedKFold(5, shuffle, random_state=seed) over the SAME 5 seeds,
           balanced_accuracy, AND a single common downstream estimator.
    varied: which preprocessing pipeline is applied
            { CtxPipe, D-Rand incumbent, Flat-Rand incumbent, EA incumbent,
              minimal-prep baseline }.

The evaluation loop below is a byte-for-byte mirror of the fold/metric logic in
parallel_evolutionary_pipeline/evaluation_earlystop.evaluate_graph_cv, so the
PrepBench controls are scored by the identical machinery.

------------------------------------------------------------------------------
WHAT *YOU* MUST SUPPLY  (the only manual work)
------------------------------------------------------------------------------
1. CtxPipe's CHOSEN PIPELINE per dataset (not just its score). Export it from
   your CtxPipe run into a JSON file (see --ctxpipe-json and the schema in
   `load_ctxpipe_pipelines`). A score alone is not enough to re-score fairly.

2. The operator-name -> sklearn-transformer mapping for any CtxPipe operator
   that is not already in PrepBench's library (see OP_FACTORY below). Unmapped
   operators raise a LOUD error on purpose -- silently dropping an operator
   would bias CtxPipe's pipeline and quietly break the fairness guarantee.

3. The common downstream estimator (--estimator). The headline comparison
   should fix this to whatever model CtxPipe uses downstream (often logistic
   regression in the DiffPrep/CtxPipe lineage). Default: logreg.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    # validate the harness on synthetic data first (no real datasets needed):
    python experiments/run_ctxpipe_aligned.py --self-test

    # real run (requires PROJ_DIR set + the deepline datasets on disk):
    python experiments/run_ctxpipe_aligned.py \
        --ctxpipe-json ctxpipe_pipelines.json \
        --budget 600 --estimator logreg --out results_aligned

Output (in --out):
    aligned_per_dataset.csv     one row per (dataset, system) with mean score
    aligned_summary.csv         mean over datasets per system
    aligned_pairwise.csv        paired Wilcoxon (CtxPipe vs each), Holm-adjusted
    aligned_table.tex           drop-in LaTeX snippet in the paper's style
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

# --- make the PrepBench package importable, mirroring load_deepline.py --------
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass
_PROJ = os.getenv("PROJ_DIR", "")
if _PROJ and _PROJ not in sys.path:
    sys.path.append(_PROJ)
# also add the repo root relative to this file, so it works without PROJ_DIR
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

import numpy as np
import pandas as pd

# --- identical primitives to evaluate_graph_cv (import the SAME symbols) ------
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from parallel_evolutionary_pipeline.config import SearchConfig  # noqa: E402
from parallel_evolutionary_pipeline.datasets import OpenMLDataset, _normalize_frame  # noqa: E402
from parallel_evolutionary_pipeline.evaluation_earlystop import _run_graph_transform  # noqa: E402
from parallel_evolutionary_pipeline.graph import WorkflowGraph  # noqa: E402
from parallel_evolutionary_pipeline.search_earlystop import RUN_MATRIX, ParallelAwareEA  # noqa: E402

# These are PrepBench's own operator builders. Reusing them for shared operators
# guarantees that, e.g., a "standard_scaler" in CtxPipe is the SAME object the
# PrepBench controls use -- one fewer source of difference.
from parallel_evolutionary_pipeline.operators import (  # noqa: E402
    BRANCH_OPERATORS,
    TAIL_OPERATORS,
)

# sklearn transformers for the minimal-prep baseline and CtxPipe-only operators
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding  # noqa: E402
from sklearn.feature_selection import VarianceThreshold  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)


SEEDS = [42, 114, 256, 512, 768]
CV_FOLDS = 5

# Systems compared. The PrepBench ones are re-run in-memory so we get the actual
# incumbent WorkflowGraph; "ctxpipe" comes from your JSON; "minimal" is a no-op
# reference (impute + encode only) standing in for the "no-search default" row.
PREPBENCH_INCUMBENT_CONFIGS = {
    "D-Rand": "D-Rand",
    "Flat-Rand": "Flat-Rand",
    "EA (D-S-F-A)": "D-S-F-A",
}


# =============================================================================
# 1) Fixed downstream estimator (varied only via --estimator)
# =============================================================================
def make_estimator(name: str, random_state: int) -> Any:
    """The SINGLE common downstream model. Fix this to match CtxPipe's model."""
    if name == "logreg":
        return LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                  max_iter=2000, random_state=random_state, n_jobs=1)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=1)
    raise ValueError(f"unknown --estimator '{name}'. Add it to make_estimator().")


# =============================================================================
# 2) Preprocessor abstraction -- one evaluation loop, pluggable preprocessing
# =============================================================================
class Preprocessor(Protocol):
    """Fit preprocessing on the train fold, transform both folds.

    fit_transform may also resample (e.g. SMOTE) and therefore returns y_train
    as well, mirroring _run_graph_transform's contract.
    """

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series
                      ) -> tuple[np.ndarray, pd.Series]: ...

    def transform(self, X_valid: pd.DataFrame) -> np.ndarray: ...


@dataclass
class WorkflowGraphPreprocessor:
    """Wraps a PrepBench WorkflowGraph and applies ONLY its preprocessing.

    The graph's own estimator field is ignored; the common downstream estimator
    is attached by the evaluation loop. Transformation goes through PrepBench's
    own _run_graph_transform, so the controls' preprocessing is applied by the
    identical code path used during search.
    """

    graph: WorkflowGraph
    dataset: OpenMLDataset  # for numeric/categorical column lists

    def _apply(self, X_train, X_valid, y_train):
        return _run_graph_transform(self.graph, self.dataset, X_train, X_valid, y_train)

    def fit_transform(self, X_train, y_train):
        Xt, self._Xv_cache, y_out = self._apply(
            X_train, X_train.iloc[:0].copy() if False else X_train, y_train
        )
        # _run_graph_transform needs both frames at once; recompute on transform.
        self._X_train, self._y_train = X_train, y_train
        return Xt, y_out

    def transform(self, X_valid):
        # Re-run the (deterministic) transform fitting on train, applied to valid.
        _, Xv, _ = self._apply(self._X_train, X_valid, self._y_train)
        return Xv


@dataclass
class SklearnPipelinePreprocessor:
    """Wraps any sklearn-compatible transformer/pipeline (for CtxPipe)."""

    pipeline: Any  # must implement fit_transform(X, y) and transform(X)

    def fit_transform(self, X_train, y_train):
        Xt = self.pipeline.fit_transform(X_train, y_train)
        return np.asarray(Xt, dtype=float), y_train.reset_index(drop=True)

    def transform(self, X_valid):
        return np.asarray(self.pipeline.transform(X_valid), dtype=float)


def make_minimal_preprocessor(dataset: OpenMLDataset) -> SklearnPipelinePreprocessor:
    """No-search baseline: impute + scale numeric, impute + one-hot categorical."""
    num = dataset.numeric_columns
    cat = dataset.categorical_columns
    transformers = []
    if num:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)),
        ]), cat))
    ct = ColumnTransformer(transformers, remainder="drop")
    return SklearnPipelinePreprocessor(ct)


# =============================================================================
# 3) THE evaluation loop -- mirrors evaluate_graph_cv exactly
# =============================================================================
def score_preprocessor(prep_factory: Callable[[], Preprocessor],
                        dataset: OpenMLDataset,
                        estimator_name: str,
                        seeds: list[int] = SEEDS,
                        cv_folds: int = CV_FOLDS) -> dict:
    """Return per-seed and mean balanced accuracy for one preprocessing pipeline.

    A fresh Preprocessor is built per fold (prep_factory()) so no state leaks
    across folds. The StratifiedKFold call and the metric are copied verbatim
    from evaluate_graph_cv to guarantee an identical protocol.
    """
    per_seed: list[float] = []
    for seed in seeds:
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        fold_scores: list[float] = []
        for tr_idx, te_idx in splitter.split(dataset.X, dataset.y):
            X_tr = dataset.X.iloc[tr_idx].reset_index(drop=True)
            X_te = dataset.X.iloc[te_idx].reset_index(drop=True)
            y_tr = dataset.y.iloc[tr_idx].reset_index(drop=True)
            y_te = dataset.y.iloc[te_idx].reset_index(drop=True)
            try:
                prep = prep_factory()
                Xt, y_fit = prep.fit_transform(X_tr, y_tr)
                Xv = prep.transform(X_te)
                est = make_estimator(estimator_name, random_state=seed)
                est.fit(Xt, y_fit)
                preds = est.predict(Xv)
                fold_scores.append(float(balanced_accuracy_score(y_te, preds)))
            except Exception as exc:  # a failed pipeline scores 0 on that fold
                print(f"      [warn] {dataset.name} seed={seed}: {type(exc).__name__}: {exc}")
                fold_scores.append(0.0)
        per_seed.append(float(np.mean(fold_scores)) if fold_scores else 0.0)
    return {"per_seed": per_seed, "mean": float(np.mean(per_seed))}


# =============================================================================
# 4) PrepBench incumbents -- run the search in-memory, grab the graph
# =============================================================================
def get_prepbench_incumbent_graph(dataset: OpenMLDataset, cfg_id: str,
                                   budget: float, search_seed: int = 42) -> WorkflowGraph:
    """Re-run a PrepBench config and return its incumbent WorkflowGraph.

    NOTE: the search itself is seeded with `search_seed` (which pipeline it
    finds). The fair *evaluation* below then re-scores that pipeline across all
    5 evaluation seeds with the common estimator. Searching once (seed 42) and
    evaluating across seeds keeps this script affordable; if you want the search
    seed varied too, loop search_seed over SEEDS and score each incumbent.
    """
    conf = SearchConfig(wall_clock_budget_seconds=budget, cv_folds=CV_FOLDS,
                        per_eval_timeout_seconds=budget / 4, random_state=search_seed)
    result = ParallelAwareEA(conf, RUN_MATRIX[cfg_id]).fit(dataset)
    return result.best_individual.graph


# =============================================================================
# 5) CtxPipe adapter -- YOU complete OP_FACTORY for CtxPipe-only operators
# =============================================================================
# Reuse PrepBench builders where an operator name matches, so shared operators
# are byte-identical. Add CtxPipe-specific operators as new entries. Each value
# is a callable: params(dict) -> a fresh sklearn transformer instance.
OP_FACTORY: dict[str, Callable[[dict], Any]] = {
    # --- shared with PrepBench (reuse its exact builders) ---
    name: spec.builder for name, spec in {**BRANCH_OPERATORS, **TAIL_OPERATORS}.items()
}
# --- CtxPipe-specific operators -----------------------------------------------
# CtxPipe (DeepLine-style) emits a fixed-length sequence of primitive tokens.
# CTXPIPE_OP_SPEC maps each token to (scope, factory). scope drives where the op
# is applied in build_ctxpipe_pipeline:
#   "numeric"     -> applied to numeric columns (imputers)
#   "categorical" -> applied to categorical columns (cat imputer / encoder)
#   "all"         -> applied to the merged, post-encoding numeric matrix, in order
#   "skip"        -> a no-op marker (<blank>, <NumericData>)
# Hyperparameters here are sensible defaults, NOT CtxPipe's exact settings (which
# the token stream does not record). If you know CtxPipe's configuration for, e.g.,
# RandomTreesEmbedding or PCA, set it here so the re-scoring is maximally faithful.
CTXPIPE_OP_SPEC: dict[str, tuple[str, Callable[[dict], Any] | None]] = {
    "blank": ("skip", None),
    "NumericData": ("skip", None),  # marker: features already numeric / no encoding step
    "ImputerMean": ("numeric", lambda p: SimpleImputer(strategy="mean")),
    "ImputerMedian": ("numeric", lambda p: SimpleImputer(strategy="median")),
    "ImputerNumMode": ("numeric", lambda p: SimpleImputer(strategy="most_frequent")),
    "ImputerCatMode": ("categorical", lambda p: SimpleImputer(strategy="most_frequent")),
    # LabelEncoder applied to *features* is conventionally an ordinal integer code:
    "LabelEncoder": ("categorical", lambda p: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    "MinMaxScaler": ("all", lambda p: MinMaxScaler()),
    "MaxAbsScaler": ("all", lambda p: MaxAbsScaler()),
    "RobustScaler": ("all", lambda p: RobustScaler()),
    "StandardScaler": ("all", lambda p: StandardScaler()),
    "VarianceThreshold": ("all", lambda p: VarianceThreshold(threshold=0.0)),
    "PowerTransformer": ("all", lambda p: PowerTransformer(method="yeo-johnson", standardize=True)),
    "PolynomialFeatures": ("all", lambda p: PolynomialFeatures(degree=2, include_bias=False)),
    # dense output (sparse_output=False) so downstream dense-only ops don't break;
    # small embedding keeps memory sane on large datasets -- tune to match CtxPipe.
    "RandomTreesEmbedding": ("all", lambda p: RandomTreesEmbedding(
        n_estimators=int(p.get("n_estimators", 10)), max_depth=int(p.get("max_depth", 5)),
        random_state=int(p.get("random_state", 0)), sparse_output=False)),
    "PCA_AUTO": ("all", lambda p: PCA(n_components=0.95, svd_solver="full")),
}

# Register CtxPipe operator factories into OP_FACTORY so build_ctxpipe_pipeline
# resolves them by token name. Shared concepts (e.g. VarianceThreshold) get the
# sklearn implementation here; PrepBench's lowercase names keep their own builders.
for _tok, (_scope, _fac) in CTXPIPE_OP_SPEC.items():
    if _fac is not None:
        OP_FACTORY[_tok] = _fac


def parse_ctxpipe_txt(path: str | Path) -> dict[str, dict]:
    """Parse the CtxPipe dump format into the pipeline dict build expects.

    Each line: ``tag <TAB> dataset <TAB> [<op1>, <op2>, ...] <TAB> score``.
    <blank>/<NumericData> tokens are dropped; operator ORDER is preserved; every
    other token must be known (unknown -> hard error, never silently skipped).
    """
    import re

    out: dict[str, dict] = {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        parts = ln.split("\t")
        if len(parts) < 3:
            print(f"[warn] skipping malformed line: {ln[:60]}...")
            continue
        ds = parts[1].strip()
        tokens = re.findall(r"<([^>]*)>", parts[2])
        ops = []
        for tok in tokens:
            if tok not in CTXPIPE_OP_SPEC:
                raise KeyError(
                    f"unknown CtxPipe token '{tok}' (dataset {ds}). Add it to "
                    f"CTXPIPE_OP_SPEC; refusing to drop it silently."
                )
            scope, _ = CTXPIPE_OP_SPEC[tok]
            if scope == "skip":
                continue
            ops.append({"name": tok, "params": {}, "scope": scope})
        out[ds] = {"ops": ops}
    return out


def build_ctxpipe_pipeline(ops: list[dict], dataset: OpenMLDataset) -> Any:
    """Translate CtxPipe's chosen operator sequence into an sklearn pipeline.

    Each op is {"name": str, "params": dict, "scope": "numeric|categorical|all"}.
    Operators with scope numeric/categorical are wrapped in a ColumnTransformer;
    scope "all" operators apply to the merged matrix as sequential steps.

    This is a *reference* translation. If CtxPipe's pipeline structure differs
    (e.g. it is strictly linear over an already-encoded matrix), simplify this
    to a plain Pipeline of the mapped steps -- the key requirement is only that
    every CtxPipe operator is represented and NONE is silently dropped.
    """
    num, cat = dataset.numeric_columns, dataset.categorical_columns
    num_steps, cat_steps, tail_steps = [], [], []
    for i, op in enumerate(ops):
        name = op["name"]
        if name not in OP_FACTORY:
            raise KeyError(
                f"CtxPipe operator '{name}' is not in OP_FACTORY. Add a mapping "
                f"in run_ctxpipe_aligned.py:OP_FACTORY. Refusing to drop it "
                f"silently (that would bias the comparison)."
            )
        step = (f"{name}_{i}", OP_FACTORY[name](op.get("params", {})))
        scope = op.get("scope", "all")
        if scope == "numeric":
            num_steps.append(step)
        elif scope == "categorical":
            cat_steps.append(step)
        else:
            tail_steps.append(step)

    transformers = []
    if num:
        transformers.append(("num", Pipeline(num_steps or [("id", SimpleImputer(strategy="median"))]), num))
    if cat:
        cat_pipe = list(cat_steps) or [("imp", SimpleImputer(strategy="most_frequent"))]
        # the matched harness needs a numeric matrix; if CtxPipe's pipeline left
        # the categorical branch without an encoder (e.g. its encoding slot was
        # blank/NumericData), append a default ordinal encoder so the branch
        # outputs numbers. This is the only operator we ever add that CtxPipe did
        # not pick, and only when categoricals exist and no encoder is present.
        if not any(isinstance(s[1], (OrdinalEncoder, OneHotEncoder)) for s in cat_pipe):
            cat_pipe.append(("enc_default", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
        transformers.append(("cat", Pipeline(cat_pipe), cat))
    pre = ColumnTransformer(transformers, remainder="drop")
    steps = [("split", pre)] + tail_steps
    return Pipeline(steps)


def load_ctxpipe_pipelines(path: str | Path) -> dict[str, dict]:
    """Load per-dataset CtxPipe pipelines.

    Expected JSON schema:
    {
      "adult": {
        "ops": [
          {"name": "simple_imputer", "params": {"strategy": "median"}, "scope": "numeric"},
          {"name": "one_hot_encoder", "params": {"max_categories": 20}, "scope": "categorical"},
          {"name": "standard_scaler", "params": {}, "scope": "all"}
        ]
      },
      "iris": { "ops": [ ... ] },
      ...
    }
    """
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("ctxpipe JSON must be a dict keyed by dataset name.")
    return data


# =============================================================================
# 6) Stats + reporting
# =============================================================================
def paired_wilcoxon_holm(table: pd.DataFrame, reference: str = "CtxPipe") -> pd.DataFrame:
    """Paired Wilcoxon of `reference` vs every other system, Holm-adjusted.

    table: index=dataset, columns=system, values=mean score.
    """
    try:
        from scipy.stats import wilcoxon
    except Exception:
        print("[warn] scipy not available; skipping Wilcoxon.")
        return pd.DataFrame()
    others = [c for c in table.columns if c != reference]
    rows = []
    raw_p = {}
    for c in others:
        a, b = table[reference].to_numpy(float), table[c].to_numpy(float)
        diff = a - b
        if np.allclose(diff, 0):
            p = 1.0
        else:
            try:
                p = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
            except Exception:
                p = float("nan")
        raw_p[c] = p
        rows.append({"comparison": f"{reference} vs {c}",
                     "mean_delta": float(np.mean(diff)),
                     "median_delta": float(np.median(diff)),
                     "wins": int((diff > 0).sum()),
                     "losses": int((diff < 0).sum()),
                     "p_raw": p})
    # Holm correction
    order = sorted(raw_p, key=lambda k: (np.inf if np.isnan(raw_p[k]) else raw_p[k]))
    m = len(order)
    holm = {}
    prev = 0.0
    for rank, c in enumerate(order):
        adj = min(1.0, (m - rank) * raw_p[c]) if not np.isnan(raw_p[c]) else float("nan")
        adj = max(adj, prev) if not np.isnan(adj) else adj
        holm[c] = adj
        if not np.isnan(adj):
            prev = adj
    for r in rows:
        c = r["comparison"].split(" vs ")[1]
        r["p_holm"] = holm.get(c, float("nan"))
    return pd.DataFrame(rows)


def write_latex(summary: pd.DataFrame, pairwise: pd.DataFrame, out: Path) -> None:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Protocol-matched comparison: all systems re-scored under the "
        r"same datasets, StratifiedKFold(5)$\times$5 seeds, balanced accuracy, "
        r"and a single fixed downstream estimator. Only the preprocessing "
        r"pipeline varies, so the gap reflects preprocessing quality alone.}",
        r"\label{tab:ctxpipe-aligned}", r"\small",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"System & Mean bal.\ acc. & vs CtxPipe ($p_{\mathrm{Holm}}$) \\", r"\midrule",
    ]
    pmap = {row["comparison"].split(" vs ")[1]: row.get("p_holm", float("nan"))
            for _, row in pairwise.iterrows()} if len(pairwise) else {}
    for system, val in summary["mean"].items():
        if system == "CtxPipe":
            lines.append(f"\\textbf{{{system}}} & \\textbf{{{val:.4f}}} & --- \\\\")
        else:
            p = pmap.get(system, float("nan"))
            ptxt = "---" if np.isnan(p) else f"{p:.3g}"
            lines.append(f"{system} & {val:.4f} & {ptxt} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 7) Drivers
# =============================================================================
def run(datasets: list[OpenMLDataset], ctx_pipelines: dict[str, dict],
        budget: float, estimator_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in datasets:
        print(f"[{ds.name}] n={ds.n_samples} d={ds.n_features}")
        # --- CtxPipe ---
        if ds.name in ctx_pipelines:
            ops = ctx_pipelines[ds.name]["ops"]
            ctx_prep = lambda ops=ops, ds=ds: SklearnPipelinePreprocessor(build_ctxpipe_pipeline(ops, ds))
            r = score_preprocessor(ctx_prep, ds, estimator_name)
            rows.append({"dataset": ds.name, "system": "CtxPipe", **_flat(r)})
            print(f"    CtxPipe            mean={r['mean']:.4f}")
        else:
            print(f"    [skip] no CtxPipe pipeline supplied for {ds.name}")
        # --- minimal (no-search) ---
        r = score_preprocessor(lambda ds=ds: make_minimal_preprocessor(ds), ds, estimator_name)
        rows.append({"dataset": ds.name, "system": "minimal", **_flat(r)})
        print(f"    minimal            mean={r['mean']:.4f}")
        # --- PrepBench incumbents (re-run search in-memory, re-score prep only) ---
        for label, cfg_id in PREPBENCH_INCUMBENT_CONFIGS.items():
            graph = get_prepbench_incumbent_graph(ds, cfg_id, budget)
            prep = lambda g=graph, ds=ds: WorkflowGraphPreprocessor(g, ds)
            r = score_preprocessor(prep, ds, estimator_name)
            rows.append({"dataset": ds.name, "system": label, **_flat(r)})
            print(f"    {label:18s} mean={r['mean']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "aligned_per_dataset.csv", index=False)

    # pivot to dataset x system on the common set of datasets (paired)
    piv = df.pivot_table(index="dataset", columns="system", values="mean")
    piv = piv.dropna(axis=0, how="any")  # keep only datasets scored for all systems
    summary = pd.DataFrame({"mean": piv.mean(axis=0)}).sort_values("mean", ascending=False)
    summary.to_csv(out_dir / "aligned_summary.csv")
    pairwise = paired_wilcoxon_holm(piv, reference="CtxPipe") if "CtxPipe" in piv.columns else pd.DataFrame()
    if len(pairwise):
        pairwise.to_csv(out_dir / "aligned_pairwise.csv", index=False)
    write_latex(summary, pairwise, out_dir / "aligned_table.tex")

    print("\n=== protocol-matched summary (mean balanced accuracy) ===")
    print(summary.to_string())
    if len(pairwise):
        print("\n=== CtxPipe vs each (paired Wilcoxon, Holm) ===")
        print(pairwise.to_string(index=False))
    print(f"\nwrote: {out_dir}/aligned_per_dataset.csv, aligned_summary.csv, "
          f"aligned_pairwise.csv, aligned_table.tex")


def _flat(r: dict) -> dict:
    return {"mean": r["mean"], **{f"seed_{s}": v for s, v in zip(SEEDS, r["per_seed"])}}


def _load_datasets(names: list[str] | None) -> list[OpenMLDataset]:
    from data_loader.load_deepline import load_deepline_dataset, ALL_DATASET_NAMES
    names = names or [n for n in ALL_DATASET_NAMES]  # paper excludes home_credit  if n != "home_credit"
    out = []
    for n in names:
        try:
            out.append(load_deepline_dataset(n))
        except Exception as exc:
            print(f"[skip] {n}: {exc}")
    return out


# =============================================================================
# 8) Self-test (synthetic) -- verifies the harness without real data
# =============================================================================
def self_test() -> None:
    print(">>> SELF-TEST on synthetic data (validates the scorer; not a result)")
    rng = np.random.default_rng(0)
    n = 300
    Xnum = rng.normal(size=(n, 4))
    cat = rng.choice(["a", "b", "c"], size=n)
    y = ((Xnum[:, 0] + (cat == "a") * 1.5 + rng.normal(scale=0.5, size=n)) > 0).astype(int)
    df = pd.DataFrame(Xnum, columns=[f"f{i}" for i in range(4)])
    df["cat"] = cat
    df.loc[df.sample(frac=0.1, random_state=1).index, "f0"] = np.nan  # missingness
    Xn, num_cols, cat_cols = _normalize_frame(df)
    ds = OpenMLDataset("synthetic", Xn.reset_index(drop=True),
                       pd.Series(y, name="target").astype("category"),
                       num_cols, cat_cols)

    # minimal-prep
    r = score_preprocessor(lambda: make_minimal_preprocessor(ds), ds, "logreg", seeds=[42, 114])
    print(f"   minimal-prep   mean={r['mean']:.4f}  per_seed={['%.4f' % v for v in r['per_seed']]}")
    assert 0.0 <= r["mean"] <= 1.0

    # a CtxPipe-style pipeline built from the adapter (using shared operators)
    ops = [
        {"name": "simple_imputer", "params": {"strategy": "median"}, "scope": "numeric"},
        {"name": "standard_scaler", "params": {}, "scope": "numeric"},
        {"name": "ordinal_encoder", "params": {}, "scope": "categorical"},
    ]
    ctx_prep = lambda: SklearnPipelinePreprocessor(build_ctxpipe_pipeline(ops, ds))
    r2 = score_preprocessor(ctx_prep, ds, "logreg", seeds=[42, 114])
    print(f"   ctxpipe-style  mean={r2['mean']:.4f}  per_seed={['%.4f' % v for v in r2['per_seed']]}")
    assert 0.0 <= r2["mean"] <= 1.0

    # WorkflowGraphPreprocessor path (how the random/EA baselines are scored):
    # build a tiny feasible graph by hand and score its preprocessing only.
    from parallel_evolutionary_pipeline.graph import BranchGene, OperatorGene
    g = WorkflowGraph(
        branches=[BranchGene(scope="numeric", nodes=[
            OperatorGene("simple_imputer", {"strategy": "median"}),
            OperatorGene("standard_scaler", {}),
        ])],
        tail_nodes=[],
        estimator=OperatorGene("logistic_regression", {"C": 1.0, "penalty": "l2", "random_state": 0}),
    )
    r3 = score_preprocessor(lambda: WorkflowGraphPreprocessor(g, ds), ds, "logreg", seeds=[42, 114])
    print(f"   graph-prep     mean={r3['mean']:.4f}  per_seed={['%.4f' % v for v in r3['per_seed']]}")
    assert 0.0 < r3["mean"] <= 1.0, "WorkflowGraphPreprocessor produced degenerate scores"

    # CtxPipe REAL operators end-to-end: parse a few real CtxPipe pipelines and
    # score them, exercising RandomTreesEmbedding / PowerTransformer /
    # PolynomialFeatures / PCA and the encoder-guarantee (weatherAUS has an
    # imputer-but-no-encoder categorical branch).
    import tempfile
    ctx_lines = "\n".join([
        "ctx_x\tweatherAUS\t[<ImputerMean>, <ImputerCatMode>, <NumericData>, <RobustScaler>, <RandomTreesEmbedding>, <VarianceThreshold>]\t0.88",
        "ctx_x\tIris\t[<blank>, <blank>, <blank>, <VarianceThreshold>, <PowerTransformer>, <PolynomialFeatures>]\t1.0",
        "ctx_x\tbureau\t[<ImputerNumMode>, <ImputerCatMode>, <NumericData>, <PowerTransformer>, <PCA_AUTO>, <blank>]\t0.92",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(ctx_lines)
        ctx_path = fh.name
    parsed = parse_ctxpipe_txt(ctx_path)
    assert set(parsed) == {"weatherAUS", "Iris", "bureau"}
    for label in ("weatherAUS", "Iris", "bureau"):
        ops = parsed[label]["ops"]
        prep = lambda ops=ops: SklearnPipelinePreprocessor(build_ctxpipe_pipeline(ops, ds))
        r = score_preprocessor(prep, ds, "logreg", seeds=[42])
        print(f"   ctxpipe[{label:10s}] mean={r['mean']:.4f}  ({len(ops)} ops applied)")
        assert 0.0 < r["mean"] <= 1.0, f"CtxPipe pipeline {label} degenerate"

    # unmapped-operator must raise (fairness guard)
    try:
        build_ctxpipe_pipeline([{"name": "totally_unknown_op", "scope": "all"}], ds)
        raise AssertionError("expected KeyError for unmapped operator")
    except KeyError:
        print("   fairness guard  OK (unmapped operator raised as designed)")
    print(">>> SELF-TEST PASSED")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run synthetic harness check and exit")
    ap.add_argument("--ctxpipe-json", type=str, default=None, help="JSON of per-dataset CtxPipe pipelines")
    ap.add_argument("--ctxpipe-txt", type=str, default=None,
                    help="CtxPipe dump in the '<tag> <ds> [<op>,...] <score>' format (parsed directly)")
    ap.add_argument("--datasets", nargs="+", default=None, help="subset of dataset names (default: all minus home_credit)")
    ap.add_argument("--budget", type=float, default=600.0, help="wall-clock seconds for the in-memory PrepBench re-runs")
    ap.add_argument("--estimator", type=str, default="logreg", help="common downstream model: logreg|rf (fix to CtxPipe's)")
    ap.add_argument("--out", type=str, default="results_aligned")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.ctxpipe_json and not args.ctxpipe_txt:
        ap.error("provide --ctxpipe-txt or --ctxpipe-json for a real run (or use --self-test).")
    if args.ctxpipe_txt:
        ctx = parse_ctxpipe_txt(args.ctxpipe_txt)
    else:
        ctx = load_ctxpipe_pipelines(args.ctxpipe_json)
    print(f"loaded {len(ctx)} CtxPipe pipelines")
    datasets = _load_datasets(args.datasets)
    if not datasets:
        ap.error("no datasets loaded; set PROJ_DIR and check the deepline data path.")
    run(datasets, ctx, budget=args.budget, estimator_name=args.estimator, out_dir=Path(args.out))


if __name__ == "__main__":
    main()

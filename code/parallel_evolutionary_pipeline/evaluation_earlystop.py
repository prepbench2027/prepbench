from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import signal
import threading
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import SearchConfig
from .datasets import OpenMLDataset
from .graph import BranchGene, OperatorGene, WorkflowGraph
from .operators import BRANCH_OPERATORS, ESTIMATOR_OPERATORS, TAIL_OPERATORS, ensure_dataframe


@contextmanager
def time_limit(seconds):
    """Best-effort per-evaluation timeout via SIGALRM.

    Works on Unix in the main thread only; it is a no-op when ``seconds`` is
    falsy/non-positive or when called off the main thread (so multi-threaded
    callers degrade gracefully to "no timeout" rather than crashing).
    """
    if not seconds or seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"evaluation exceeded {seconds:.1f}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# ── Subprocess hard-timeout helpers ──────────────────────────────────────
# These are module-level so they are picklable for multiprocessing spawn.


def _cv_eval_worker(graph, dataset, config, result_queue):
    """Run evaluate_graph_cv in a subprocess (module-level for pickling)."""
    try:
        result = evaluate_graph_cv(graph, dataset, config)
        result_queue.put(result)
    except Exception:
        result_queue.put(None)


def _screen_eval_worker(graph, dataset, config, result_queue):
    """Run evaluate_graph_screen in a subprocess (module-level for pickling)."""
    try:
        result = evaluate_graph_screen(graph, dataset, config)
        result_queue.put(result)
    except Exception:
        result_queue.put(None)


def _run_with_hard_timeout(worker_fn, graph, dataset, config, timeout):
    """Run an evaluation function in a subprocess with a hard kill timeout.

    This is the only reliable way to interrupt C extensions (scikit-learn
    .fit() calls) that ignore SIGALRM.  The worker process is hard-killed
    after *timeout* seconds, guaranteeing no frozen search loop.
    """
    queue = mp.get_context("spawn").Queue()
    proc = mp.get_context("spawn").Process(
        target=worker_fn, args=(graph, dataset, config, queue)
    )
    start = time.perf_counter()
    proc.start()
    proc.join(timeout=timeout)

    runtime = time.perf_counter() - start

    if proc.is_alive():
        proc.terminate()
        proc.join()
        cost = estimate_cost(graph, dataset, config)
        return EvaluationResult(
            score=0.0, runtime_seconds=runtime, cost=cost, status="timeout"
        )

    try:
        result = queue.get_nowait()
    except Exception:
        result = None

    if result is None:
        cost = estimate_cost(graph, dataset, config)
        return EvaluationResult(
            score=0.0, runtime_seconds=runtime, cost=cost, status="failed"
        )

    # Record actual wall time including subprocess spawn overhead.
    result.runtime_seconds = runtime
    return result


@dataclass
class CostEstimate:
    critical_path_cost: float
    synchronization_cost: float
    merge_cost: float
    fit_cost: float
    total_cost: float
    serial_cost: float


@dataclass
class EvaluationResult:
    score: float
    runtime_seconds: float
    cost: CostEstimate
    status: str = "ok"


def _subset_frame(dataset: OpenMLDataset, X: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "numeric":
        return X[dataset.numeric_columns].copy()
    if scope == "categorical":
        return X[dataset.categorical_columns].copy()
    return X.copy()


def _fit_transform_gene(gene: OperatorGene, X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = BRANCH_OPERATORS.get(gene.operator) or TAIL_OPERATORS.get(gene.operator)
    if spec is None:
        raise ValueError(f"Unknown operator: {gene.operator}")
    params = dict(gene.params)
    if gene.operator == "simple_imputer" and not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_train.dtypes):
        if params.get("strategy") in {"mean", "median"}:
            params["strategy"] = "most_frequent"
    if gene.operator == "select_k_best":
        params["k"] = max(1, min(int(params["k"]), X_train.shape[1]))
    component = spec.builder(params)
    if gene.operator == "smote":
        raise ValueError("SMOTE should be handled in tail resampling, not via fit_transform.")
    if spec.requires_y:
        Xt = component.fit_transform(X_train, y_train)
    else:
        Xt = component.fit_transform(X_train)
    Xv = component.transform(X_valid)
    return ensure_dataframe(Xt, gene.operator), ensure_dataframe(Xv, gene.operator)


def _run_branch(branch: BranchGene, dataset: OpenMLDataset, X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    Xt = _subset_frame(dataset, X_train, branch.scope)
    Xv = _subset_frame(dataset, X_valid, branch.scope)
    for gene in branch.nodes:
        Xt, Xv = _fit_transform_gene(gene, Xt, Xv, y_train)
    return Xt, Xv


def _run_graph_transform(graph: WorkflowGraph, dataset: OpenMLDataset, X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    for branch in graph.branches:
        Xt, Xv = _run_branch(branch, dataset, X_train, X_valid, y_train)
        train_parts.append(Xt)
        valid_parts.append(Xv)
    merged_train = pd.concat(train_parts, axis=1).reset_index(drop=True)
    merged_valid = pd.concat(valid_parts, axis=1).reset_index(drop=True)
    y_out = y_train.reset_index(drop=True)
    for gene in graph.tail_nodes:
        if gene.operator == "smote":
            sampler = TAIL_OPERATORS[gene.operator].builder(gene.params)
            X_res, y_res = sampler.fit_resample(merged_train.to_numpy(dtype=float), y_out)
            merged_train = ensure_dataframe(X_res, "smote")
            y_out = pd.Series(y_res)
        else:
            merged_train, merged_valid = _fit_transform_gene(gene, merged_train, merged_valid, y_out)
    return merged_train.to_numpy(dtype=float), merged_valid.to_numpy(dtype=float), y_out


def estimate_cost(graph: WorkflowGraph, dataset: OpenMLDataset, config: SearchConfig) -> CostEstimate:
    n = max(1, dataset.n_samples)
    d = max(1, dataset.n_features)
    branch_costs: list[float] = []
    for branch in graph.branches:
        branch_features = len(dataset.numeric_columns) if branch.scope == "numeric" else (
            len(dataset.categorical_columns) if branch.scope == "categorical" else d
        )
        cost = 0.0
        for gene in branch.nodes:
            spec = BRANCH_OPERATORS[gene.operator]
            cost += spec.relative_cost * np.log1p(n) * np.log1p(max(1, branch_features))
        branch_costs.append(cost)
    tail_cost = 0.0
    for gene in graph.tail_nodes:
        spec = TAIL_OPERATORS[gene.operator]
        tail_cost += spec.relative_cost * np.log1p(n) * np.log1p(d)
    fit_spec = ESTIMATOR_OPERATORS[graph.estimator.operator]
    fit_cost = fit_spec.relative_cost * np.log1p(n) * np.sqrt(d)
    critical_path = (max(branch_costs) if branch_costs else 0.0) + tail_cost
    synchronization = 0.0 if len(branch_costs) <= 1 else config.cost_alpha * (max(branch_costs) - min(branch_costs))
    merge_cost = config.cost_beta * len(graph.branches) * d
    serial_cost = sum(branch_costs) + tail_cost + merge_cost + fit_cost
    total_cost = critical_path + synchronization + merge_cost + fit_cost
    return CostEstimate(
        critical_path_cost=float(critical_path),
        synchronization_cost=float(synchronization),
        merge_cost=float(merge_cost),
        fit_cost=float(fit_cost),
        total_cost=float(total_cost),
        serial_cost=float(serial_cost),
    )


def evaluate_graph_cv(graph: WorkflowGraph, dataset: OpenMLDataset, config: SearchConfig,
                      hard_timeout: float | None = None) -> EvaluationResult:
    # Hard subprocess timeout for C extensions that ignore SIGALRM.
    if hard_timeout is not None and hard_timeout > 0:
        return _run_with_hard_timeout(_cv_eval_worker, graph, dataset, config, hard_timeout)

    start = time.perf_counter()
    cost = estimate_cost(graph, dataset, config)
    splitter = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    scores: list[float] = []
    try:
        with time_limit(config.per_eval_timeout_seconds):
            for train_idx, test_idx in splitter.split(dataset.X, dataset.y):
                # belt-and-suspenders: stop this evaluation mid-way if per-eval timeout is exceeded
                if config.per_eval_timeout_seconds and time.perf_counter() - start >= config.per_eval_timeout_seconds:
                    break
                X_train = dataset.X.iloc[train_idx].reset_index(drop=True)
                X_test = dataset.X.iloc[test_idx].reset_index(drop=True)
                y_train = dataset.y.iloc[train_idx].reset_index(drop=True)
                y_test = dataset.y.iloc[test_idx].reset_index(drop=True)
                Xt, Xv, y_fit = _run_graph_transform(graph, dataset, X_train, X_test, y_train)
                estimator = ESTIMATOR_OPERATORS[graph.estimator.operator].builder(graph.estimator.params)
                estimator.fit(Xt, y_fit)
                preds = estimator.predict(Xv)
                scores.append(float(balanced_accuracy_score(y_test, preds)))
        if scores:
            score = float(np.mean(scores))
            status = "ok"
        else:
            score = 0.0
            status = "timeout"
    except TimeoutError:
        score = 0.0
        status = "timeout"
        print(f'[{dataset}] timeout')
    except Exception:
        score = 0.0
        status = "failed"
    return EvaluationResult(score=score, runtime_seconds=time.perf_counter() - start, cost=cost, status=status)


def evaluate_graph_screen(graph: WorkflowGraph, dataset: OpenMLDataset, config: SearchConfig,
                          hard_timeout: float | None = None) -> EvaluationResult:
    # Hard subprocess timeout for C extensions that ignore SIGALRM.
    if hard_timeout is not None and hard_timeout > 0:
        return _run_with_hard_timeout(_screen_eval_worker, graph, dataset, config, hard_timeout)

    start = time.perf_counter()
    cost = estimate_cost(graph, dataset, config)
    try:
        with time_limit(config.per_eval_timeout_seconds):
            X_small, _, y_small, _ = train_test_split(
                dataset.X,
                dataset.y,
                train_size=config.stage1_subsample_fraction,
                stratify=dataset.y,
                random_state=config.random_state,
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X_small,
                y_small,
                test_size=config.screening_test_size,
                stratify=y_small,
                random_state=config.random_state,
            )
            Xt, Xv, y_fit = _run_graph_transform(
                graph,
                OpenMLDataset(dataset.name, X_small.reset_index(drop=True), y_small.reset_index(drop=True), dataset.numeric_columns, dataset.categorical_columns),
                X_train.reset_index(drop=True),
                X_test.reset_index(drop=True),
                y_train.reset_index(drop=True),
            )
            estimator = ESTIMATOR_OPERATORS[graph.estimator.operator].builder(graph.estimator.params)
            estimator.fit(Xt, y_fit)
            preds = estimator.predict(Xv)
            score = float(balanced_accuracy_score(y_test.reset_index(drop=True), preds))
        status = "ok"
    except TimeoutError:
        score = 0.0
        status = "timeout"
    except Exception:
        score = 0.0
        status = "failed"
    return EvaluationResult(score=score, runtime_seconds=time.perf_counter() - start, cost=cost, status=status)

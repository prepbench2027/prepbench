from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomTreesEmbedding
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.svm import SVC


class TargetMeanEncoder:
    def __init__(self, smoothing: float = 5.0):
        self.smoothing = float(smoothing)
        self.global_mean_: float = 0.0
        self.mapping_: dict[str, dict[Any, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetMeanEncoder":
        self.global_mean_ = float(pd.Series(y).astype("category").cat.codes.mean())
        y_codes = pd.Series(y).astype("category").cat.codes
        self.mapping_ = {}
        for col in X.columns:
            stats = pd.DataFrame({"x": X[col].astype("object"), "y": y_codes}).groupby("x")["y"].agg(["mean", "count"])
            smoothed = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (stats["count"] + self.smoothing)
            self.mapping_[col] = smoothed.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            mapping = self.mapping_.get(col, {})
            out[col] = X[col].astype("object").map(mapping).fillna(self.global_mean_).astype(float)
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class SimpleSMOTE:
    def __init__(self, k_neighbors: int = 5, random_state: int = 0):
        self.k_neighbors = int(k_neighbors)
        self.random_state = int(random_state)

    def fit_resample(self, X: np.ndarray, y: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        labels, counts = np.unique(y_arr, return_counts=True)
        max_count = int(counts.max())
        X_parts = [X_arr]
        y_parts = [y_arr]
        for label, count in zip(labels, counts):
            if count >= max_count:
                continue
            minority = X_arr[y_arr == label]
            if len(minority) < 2:
                continue
            n_neighbors = max(1, min(self.k_neighbors, len(minority) - 1))
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=1)
            nn.fit(minority)
            needed = max_count - len(minority)
            samples = []
            for _ in range(needed):
                idx = int(rng.integers(0, len(minority)))
                neighbors = nn.kneighbors(minority[[idx]], return_distance=False)[0][1:]
                nn_idx = int(rng.choice(neighbors))
                alpha = float(rng.random())
                samples.append(minority[idx] + alpha * (minority[nn_idx] - minority[idx]))
            if samples:
                synth = np.vstack(samples)
                X_parts.append(synth)
                y_parts.append(np.full(len(samples), label, dtype=y_arr.dtype))
        return np.vstack(X_parts), np.concatenate(y_parts)


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    category: str
    stage: str
    allowed_inputs: tuple[str, ...]
    output_signature: str
    builder: Callable[[dict[str, Any]], Any]
    default_params: dict[str, Any] = field(default_factory=dict)
    param_space: dict[str, list[Any]] = field(default_factory=dict)
    relative_cost: float = 1.0
    requires_y: bool = False


BRANCH_OPERATORS: dict[str, OperatorSpec] = {
    "simple_imputer": OperatorSpec(
        "simple_imputer", "imputation", "branch", ("numeric", "categorical", "mixed"), "same",
        lambda p: SimpleImputer(strategy=p["strategy"]),
        default_params={"strategy": "most_frequent"},
        param_space={"strategy": ["mean", "median", "most_frequent"]},
        relative_cost=0.8,
    ),
    "knn_imputer": OperatorSpec(
        "knn_imputer", "imputation", "branch", ("numeric",), "numeric",
        lambda p: KNNImputer(n_neighbors=int(p["k"])),
        default_params={"k": 5},
        param_space={"k": [3, 5, 7]},
        relative_cost=2.0,
    ),
    "standard_scaler": OperatorSpec(
        "standard_scaler", "scaling", "branch", ("numeric",), "numeric",
        lambda p: StandardScaler(),
        relative_cost=0.8,
    ),
    "minmax_scaler": OperatorSpec(
        "minmax_scaler", "scaling", "branch", ("numeric",), "numeric",
        lambda p: MinMaxScaler(),
        relative_cost=0.8,
    ),
    "robust_scaler": OperatorSpec(
        "robust_scaler", "scaling", "branch", ("numeric",), "numeric",
        lambda p: RobustScaler(quantile_range=tuple(p["quantile_range"])),
        default_params={"quantile_range": (25.0, 75.0)},
        param_space={"quantile_range": [(10.0, 90.0), (25.0, 75.0), (30.0, 70.0)]},
        relative_cost=1.0,
    ),
    "one_hot_encoder": OperatorSpec(
        "one_hot_encoder", "encoding", "branch", ("categorical", "mixed"), "numeric",
        lambda p: OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=p["max_categories"]),
        default_params={"max_categories": None},
        param_space={"max_categories": [None, 10, 20]},
        relative_cost=1.5,
    ),
    "ordinal_encoder": OperatorSpec(
        "ordinal_encoder", "encoding", "branch", ("categorical", "mixed"), "numeric",
        lambda p: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        relative_cost=1.0,
    ),
    "target_encoder": OperatorSpec(
        "target_encoder", "encoding", "branch", ("categorical", "mixed"), "numeric",
        lambda p: TargetMeanEncoder(smoothing=float(p["smoothing"])),
        default_params={"smoothing": 5.0},
        param_space={"smoothing": [2.0, 5.0, 10.0]},
        relative_cost=1.6,
        requires_y=True,
    ),
    "select_k_best": OperatorSpec(
        "select_k_best", "feature_selection", "both", ("numeric",), "numeric",
        lambda p: SelectKBest(score_func=f_classif if p["score_func"] == "f_classif" else mutual_info_classif, k=int(p["k"])),
        default_params={"k": 10, "score_func": "f_classif"},
        param_space={"k": [5, 10, 20, 50], "score_func": ["f_classif", "mutual_info"]},
        relative_cost=1.3,
        requires_y=True,
    ),
    "variance_threshold": OperatorSpec(
        "variance_threshold", "feature_selection", "both", ("numeric",), "numeric",
        lambda p: VarianceThreshold(threshold=float(p["threshold"])),
        default_params={"threshold": 0.0},
        param_space={"threshold": [0.0, 0.001, 0.01]},
        relative_cost=0.7,
    ),
    "polynomial_features": OperatorSpec(
        "polynomial_features", "feature_engineering", "both", ("numeric",), "numeric",
        lambda p: PolynomialFeatures(),
        default_params={},
        param_space={},
        relative_cost=1.7
    ),
    "random_trees_embedding": OperatorSpec(
        "random_trees_embedding", "feature_engineering", "both", ("numeric",), "numeric",
        lambda p: RandomTreesEmbedding(random_state=0),
        default_params={},
        param_space={},
        relative_cost=2.0
    ),
    "pca_auto": OperatorSpec(
        "pca_auto", "feature_engineering", "both", ("numeric",), "numeric",
        lambda p: PCA(svd_solver="auto"),
        default_params={},
        param_space={},
        relative_cost=1.5
    )
}

TAIL_OPERATORS: dict[str, OperatorSpec] = {
    name: spec for name, spec in BRANCH_OPERATORS.items() if spec.stage == "both"
}
TAIL_OPERATORS["smote"] = OperatorSpec(
    "smote", "resampling", "tail", ("numeric",), "numeric",
    lambda p: SimpleSMOTE(k_neighbors=int(p["k_neighbors"]), random_state=int(p["random_state"])),
    default_params={"k_neighbors": 5, "random_state": 0},
    param_space={"k_neighbors": [3, 5, 7]},
    relative_cost=2.2,
)

ESTIMATOR_OPERATORS: dict[str, OperatorSpec] = {
    "random_forest": OperatorSpec(
        "random_forest", "estimator", "estimator", ("numeric",), "numeric",
        lambda p: RandomForestClassifier(
            n_estimators=int(p["n_estimators"]),
            max_depth=None if p["max_depth"] is None else int(p["max_depth"]),
            random_state=int(p["random_state"]),
            n_jobs=1,
        ),
        default_params={"n_estimators": 200, "max_depth": None, "random_state": 0},
        param_space={"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10, 20]},
        relative_cost=3.0,
    ),
    "gradient_boosting": OperatorSpec(
        "gradient_boosting", "estimator", "estimator", ("numeric",), "numeric",
        lambda p: GradientBoostingClassifier(
            n_estimators=int(p["n_estimators"]),
            learning_rate=float(p["learning_rate"]),
            max_depth=int(p["max_depth"]),
            random_state=int(p["random_state"]),
        ),
        default_params={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 0},
        param_space={"n_estimators": [50, 100], "learning_rate": [0.03, 0.1], "max_depth": [3, 5]},
        relative_cost=3.5,
    ),
    "logistic_regression": OperatorSpec(
        "logistic_regression", "estimator", "estimator", ("numeric",), "numeric",
        lambda p: LogisticRegression(
            C=float(p["C"]),
            penalty=str(p["penalty"]),
            solver="liblinear" if p["penalty"] == "l1" else "lbfgs",
            max_iter=2000,
            random_state=int(p["random_state"]),
            n_jobs=1,
        ),
        default_params={"C": 1.0, "penalty": "l2", "random_state": 0},
        param_space={"C": [0.1, 1.0, 10.0], "penalty": ["l1", "l2"]},
        relative_cost=1.5,
    ),
    "svm_rbf": OperatorSpec(
        "svm_rbf", "estimator", "estimator", ("numeric",), "numeric",
        lambda p: SVC(C=float(p["C"]), gamma=p["gamma"], kernel="rbf", max_iter=500, tol=float(p["tol"])),
        default_params={"C": 1.0, "gamma": "scale", "tol": 0.001},
        param_space={"C": [0.3, 1.0, 3.0], "gamma": ["scale", "auto"], "tol": [0.001, 0.01]},
        relative_cost=4.0,
    ),
}


def ensure_dataframe(X: Any, prefix: str) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    if isinstance(X, pd.Series):
        return X.to_frame().reset_index(drop=True)
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cols = [f"{prefix}_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


OPENML_DATASET_NAMES = [
    "abalone",
    "ada_prior",
    # "avila",
    # "connect-4",
    # "eeg",
    # "google",
    # "house",
    # "jungle_chess",
    # "micro",
    # "mozilla4",
    # "obesity",
    # "page-blocks",
    # "pcseq",
    # "pol",
    # "run_or_walk",
    # "shuttle",
    # "uscensus",
    # "wall-robot",
]


@dataclass
class OpenMLDataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    numeric_columns: list[str]
    categorical_columns: list[str]

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


def _normalize_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    X = frame.copy()
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]
    for col in categorical_columns:
        X[col] = X[col].astype("category")
    return X, numeric_columns, categorical_columns


def load_openml_dataset(name: str, data_home: str | Path | None = None) -> OpenMLDataset:
    fetched = fetch_openml(
        name=name,
        version="active",
        as_frame=True,
        data_home=None if data_home is None else str(data_home),
    )
    X = fetched.data if isinstance(fetched.data, pd.DataFrame) else pd.DataFrame(fetched.data)
    y = fetched.target if isinstance(fetched.target, pd.Series) else pd.Series(fetched.target, name="target")
    X, numeric_columns, categorical_columns = _normalize_frame(X)
    y = y.astype("category")
    return OpenMLDataset(
        name=name,
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

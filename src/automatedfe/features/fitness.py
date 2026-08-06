"""Fitness evaluation for generated transaction features."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .feature_materialization import FeatureMaterializer
from .feature_spec import FeatureSpec

DATASET_MERCHANT_COLUMN = "merchant_id"
DATASET_TIMESTAMP_COLUMN = "event_timestamp"
DATASET_TARGET_COLUMN = "label"
DATASET_SPLIT_COLUMN = "split"
TRAIN_SPLIT = "train"
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_ITERATIONS = 1_000


def _load_ordered_events(dataset_path: str | PathLike[str]) -> tuple[np.ndarray, ...]:
    """Load the training events once, in chronological order."""

    path = Path(dataset_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset parquet file does not exist: {path}")

    events = duckdb.sql(
        """
        SELECT
            merchant_id,
            epoch_us(event_timestamp) AS event_timestamp,
            label
        FROM read_parquet(?)
        WHERE split = ?
        ORDER BY event_timestamp
        """,
        params=[str(path), TRAIN_SPLIT],
    ).fetchnumpy()

    merchants = np.asarray(events[DATASET_MERCHANT_COLUMN])
    timestamps = np.asarray(events[DATASET_TIMESTAMP_COLUMN], dtype=np.int64)
    labels = np.asarray(events[DATASET_TARGET_COLUMN])
    if not len(labels):
        raise ValueError("Dataset has no rows in the training split")
    if np.unique(labels).size < 2:
        raise ValueError("Fitness requires at least two target classes")
    return merchants, timestamps, labels


class LogisticRegressionFitness:
    """Materialize one feature, fit on train, and score on validation."""

    def __init__(
        self,
        materializer: FeatureMaterializer,
        dataset_path: str | PathLike[str],
        *,
        validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
        score_metric: str = "roc_auc",
        random_state: int = DEFAULT_RANDOM_STATE,
        max_iter: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if score_metric not in {"accuracy", "roc_auc"}:
            raise ValueError("score_metric must be 'accuracy' or 'roc_auc'")

        self.materializer = materializer
        self.dataset_path = Path(dataset_path).resolve()
        self.validation_fraction = validation_fraction
        self.score_metric = score_metric
        self.random_state = random_state
        self.max_iter = max_iter
        self.event_merchants, self.event_timestamps, self.labels = _load_ordered_events(
            self.dataset_path
        )
        self.last_model: LogisticRegression | None = None

        validation_rows = max(1, int(np.ceil(len(self.labels) * validation_fraction)))
        split = len(self.labels) - validation_rows
        self.fit_indices = np.arange(split)
        self.validation_indices = np.arange(split, len(self.labels))
        if split == 0 or np.unique(self.labels[self.fit_indices]).size < 2:
            raise ValueError("Chronological fit rows must contain at least two target classes")

    def _values_for(self, individual: FeatureSpec | Any) -> np.ndarray:
        return self.materializer.materialize_for_events(
            individual,
            self.event_merchants,
            self.event_timestamps,
        )

    def prepare_population(self, individuals: Sequence[FeatureSpec | Any]) -> None:
        """Calculate and cache every feature before population evaluation."""

        for individual in individuals:
            self._values_for(individual)

    def __call__(self, individual: FeatureSpec | Any) -> float:
        values = self._values_for(individual).reshape(-1, 1)
        x_train = values[self.fit_indices]
        x_validation = values[self.validation_indices]
        y_train = self.labels[self.fit_indices]
        y_validation = self.labels[self.validation_indices]

        imputer = SimpleImputer(strategy="constant", fill_value=0.0)
        x_train = imputer.fit_transform(x_train)
        x_validation = imputer.transform(x_validation)

        model = LogisticRegression(max_iter=self.max_iter, random_state=self.random_state)
        model.fit(x_train, y_train)
        self.last_model = model

        if self.score_metric == "accuracy":
            return float(model.score(x_validation, y_validation))
        if np.unique(y_validation).size < 2:
            raise ValueError("ROC AUC requires both target classes in the validation rows")
        return float(roc_auc_score(y_validation, model.predict_proba(x_validation)[:, 1]))

    evaluate = __call__


FitnessEvaluator = LogisticRegressionFitness

__all__ = [
    "DATASET_MERCHANT_COLUMN",
    "DATASET_SPLIT_COLUMN",
    "DATASET_TARGET_COLUMN",
    "DATASET_TIMESTAMP_COLUMN",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_VALIDATION_FRACTION",
    "FitnessEvaluator",
    "LogisticRegressionFitness",
    "TRAIN_SPLIT",
]

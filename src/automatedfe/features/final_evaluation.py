"""Final evaluation of a generated feature set on the held-out test split."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from .feature_materialization import FeatureMaterializer
from .feature_types import TxFeature
from .fitness import (
    DATASET_MERCHANT_COLUMN,
    DATASET_TARGET_COLUMN,
    DATASET_TIMESTAMP_COLUMN,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_RANDOM_STATE,
    TRAIN_SPLIT,
)

TEST_SPLIT = "test"

logger = logging.getLogger(__name__)


def _load_split_events(
    dataset_path: str | PathLike[str],
    split: str,
) -> tuple[np.ndarray, ...]:
    """Load one split's events in chronological order."""

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
        ORDER BY event_timestamp, merchant_id
        """,
        params=[str(path), split],
    ).fetchnumpy()

    merchants = np.asarray(events[DATASET_MERCHANT_COLUMN])
    timestamps = np.asarray(events[DATASET_TIMESTAMP_COLUMN], dtype=np.int64)
    labels = np.asarray(events[DATASET_TARGET_COLUMN])
    return merchants, timestamps, labels


def _deduplicate_individuals(individuals: Sequence[Any]) -> list[Any]:
    seen: set[str] = set()
    unique: list[Any] = []
    for individual in individuals:
        name = (
            individual.name
            if isinstance(individual, TxFeature)
            else str(individual)
        )
        if name in seen:
            continue
        seen.add(name)
        unique.append(individual)
    return unique


@dataclass(frozen=True, slots=True)
class FinalEvaluationResult:
    """Test-split metrics and the fitted final model."""

    metrics: dict[str, float]
    model: LogisticRegression
    predictions: np.ndarray


class FinalEvaluator:
    """Materialize a generated feature set and score it on the test split.

    The complete feature matrix is computed over the union of training and
    test events, then a single logistic regression is fitted on the training
    rows and scored on the held-out test rows.
    """

    def __init__(
        self,
        materializer: FeatureMaterializer,
        dataset_path: str | PathLike[str],
        *,
        random_state: int = DEFAULT_RANDOM_STATE,
        max_iter: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        self.materializer = materializer
        self.dataset_path = Path(dataset_path).resolve()
        self.random_state = random_state
        self.max_iter = max_iter

        train_merchants, train_timestamps, train_labels = _load_split_events(
            self.dataset_path, TRAIN_SPLIT
        )
        test_merchants, test_timestamps, test_labels = _load_split_events(
            self.dataset_path, TEST_SPLIT
        )
        if not len(train_labels):
            raise ValueError("Dataset has no rows in the training split")
        if not len(test_labels):
            raise ValueError("Dataset has no rows in the test split")
        if np.unique(train_labels).size < 2:
            raise ValueError(
                "Final evaluation requires at least two target classes in the training split"
            )

        self.event_merchants = np.concatenate([train_merchants, test_merchants])
        self.event_timestamps = np.concatenate([train_timestamps, test_timestamps])
        self.labels = np.concatenate([train_labels, test_labels])
        self.train_indices = np.arange(len(train_labels))
        self.test_indices = np.arange(len(train_labels), len(self.labels))

    def _feature_matrix(self, individuals: Sequence[Any]) -> np.ndarray:
        columns = [
            self.materializer.materialize_for_events(
                individual,
                self.event_merchants,
                self.event_timestamps,
            )
            for individual in individuals
        ]
        return np.column_stack(columns)

    def evaluate(self, individuals: Sequence[Any]) -> FinalEvaluationResult:
        """Fit the final model on the training rows and score the test rows."""

        individuals = _deduplicate_individuals(individuals)
        if not individuals:
            raise ValueError("At least one individual is required for final evaluation")
        matrix = self._feature_matrix(individuals)
        x_train = matrix[self.train_indices]
        x_test = matrix[self.test_indices]
        y_train = self.labels[self.train_indices]
        y_test = self.labels[self.test_indices]

        imputer = SimpleImputer(strategy="constant", fill_value=0.0)
        x_train = imputer.fit_transform(x_train)
        x_test = imputer.transform(x_test)

        model = LogisticRegression(max_iter=self.max_iter, random_state=self.random_state)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        logger.info(
            "Final model: %d features, %d training rows, %d test rows",
            matrix.shape[1],
            len(y_train),
            len(y_test),
        )

        if np.unique(y_test).size < 2:
            raise ValueError("ROC AUC requires both target classes in the test split")
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "roc_auc": float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])),
        }
        return FinalEvaluationResult(
            metrics=metrics,
            model=model,
            predictions=predictions,
        )

    __call__ = evaluate


__all__ = [
    "TEST_SPLIT",
    "FinalEvaluationResult",
    "FinalEvaluator",
]

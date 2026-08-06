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
from sklearn.model_selection import TimeSeriesSplit

from .feature_materialization import FeatureMaterializer
from .feature_spec import FeatureSpec

DATASET_MERCHANT_COLUMN = "merchant_id"
DATASET_TIMESTAMP_COLUMN = "event_timestamp"
DATASET_TARGET_COLUMN = "label"
DATASET_SPLIT_COLUMN = "split"
TRAIN_SPLIT = "train"
DEFAULT_N_SPLITS = 3
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


def _time_series_splits(
    timestamps: np.ndarray,
    n_splits: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Build expanding time-series folds without splitting a timestamp group."""

    unique_timestamps, timestamp_groups = np.unique(timestamps, return_inverse=True)
    if len(unique_timestamps) <= n_splits:
        raise ValueError(
            "Time-series cross-validation requires more distinct timestamps than "
            f"n_splits ({n_splits})"
        )

    # Split timestamp groups, rather than individual rows.  TimeSeriesSplit does
    # not shuffle and therefore keeps every validation group strictly after its
    # corresponding training groups.  Splitting groups is essential when several
    # events have exactly the same timestamp.
    group_splitter = TimeSeriesSplit(n_splits=n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_groups, validation_groups in group_splitter.split(unique_timestamps):
        fit_indices = np.flatnonzero(np.isin(timestamp_groups, train_groups))
        validation_indices = np.flatnonzero(np.isin(timestamp_groups, validation_groups))

        if timestamps[fit_indices[-1]] >= timestamps[validation_indices[0]]:
            raise ValueError(
                "Time-series cross-validation would place a training timestamp "
                "at or after the first validation timestamp"
            )
        splits.append((fit_indices, validation_indices))

    return tuple(splits)


class LogisticRegressionFitness:
    """Materialize one feature and score it with chronological cross-validation."""

    def __init__(
        self,
        materializer: FeatureMaterializer,
        dataset_path: str | PathLike[str],
        *,
        n_splits: int = DEFAULT_N_SPLITS,
        score_metric: str = "roc_auc",
        random_state: int = DEFAULT_RANDOM_STATE,
        max_iter: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if score_metric not in {"accuracy", "roc_auc"}:
            raise ValueError("score_metric must be 'accuracy' or 'roc_auc'")

        self.materializer = materializer
        self.dataset_path = Path(dataset_path).resolve()
        self.n_splits = n_splits
        self.score_metric = score_metric
        self.random_state = random_state
        self.max_iter = max_iter
        self.event_merchants, self.event_timestamps, self.labels = _load_ordered_events(
            self.dataset_path
        )
        self.last_model: LogisticRegression | None = None
        try:
            self.cv_splits = _time_series_splits(self.event_timestamps, n_splits)
        except ValueError as error:
            if len(np.unique(self.event_timestamps)) <= n_splits:
                raise ValueError(
                    "Chronological fit rows must contain at least two target classes "
                    "and time-series cross-validation requires more distinct "
                    f"timestamps than n_splits ({n_splits})"
                ) from error
            raise

        for fit_indices, _ in self.cv_splits:
            if np.unique(self.labels[fit_indices]).size < 2:
                raise ValueError("Chronological fit rows must contain at least two target classes")

        # Keep the historical single-fold attributes pointing at the final,
        # latest fold.  ``cv_splits`` contains all folds used for scoring.
        self.fit_indices, self.validation_indices = self.cv_splits[-1]
        self.fold_scores: list[float] = []
        self.last_models: list[LogisticRegression] = []

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
        self.fold_scores = []
        self.last_models = []

        for fit_indices, validation_indices in self.cv_splits:
            x_train = values[fit_indices]
            x_validation = values[validation_indices]
            y_train = self.labels[fit_indices]
            y_validation = self.labels[validation_indices]

            imputer = SimpleImputer(strategy="constant", fill_value=0.0)
            x_train = imputer.fit_transform(x_train)
            x_validation = imputer.transform(x_validation)

            model = LogisticRegression(max_iter=self.max_iter, random_state=self.random_state)
            model.fit(x_train, y_train)
            self.last_models.append(model)

            if self.score_metric == "accuracy":
                score = float(model.score(x_validation, y_validation))
            else:
                if np.unique(y_validation).size < 2:
                    raise ValueError(
                        "ROC AUC requires both target classes in every validation fold"
                    )
                score = float(
                    roc_auc_score(y_validation, model.predict_proba(x_validation)[:, 1])
                )
            self.fold_scores.append(score)

        self.last_model = self.last_models[-1]
        return float(np.mean(self.fold_scores))

    evaluate = __call__


FitnessEvaluator = LogisticRegressionFitness

__all__ = [
    "DATASET_MERCHANT_COLUMN",
    "DATASET_SPLIT_COLUMN",
    "DATASET_TARGET_COLUMN",
    "DATASET_TIMESTAMP_COLUMN",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_N_SPLITS",
    "DEFAULT_RANDOM_STATE",
    "FitnessEvaluator",
    "LogisticRegressionFitness",
    "TRAIN_SPLIT",
]

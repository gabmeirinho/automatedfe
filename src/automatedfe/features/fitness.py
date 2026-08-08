"""Fitness evaluation for generated transaction features."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

from .feature_materialization import FeatureMaterializer

logger = logging.getLogger(__name__)

DATASET_MERCHANT_COLUMN = "merchant_id"
DATASET_TIMESTAMP_COLUMN = "event_timestamp"
DATASET_TARGET_COLUMN = "label"
DATASET_SPLIT_COLUMN = "split"
TRAIN_SPLIT = "train"
DEFAULT_N_SPLITS = 3
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_ITERATIONS = 1_000
RESIDUAL_TREE_PARAMS = {
    "max_depth": 1,
    "min_samples_leaf": 150,
    "random_state": DEFAULT_RANDOM_STATE,
}
RESIDUAL_EPSILON = 1e-6
RESIDUAL_SHRINKAGE = 0.2
MIN_LOGIT_WEIGHT = 1e-4


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    """Convert log-odds to probabilities without overflowing ``exp``."""

    lower = np.log(RESIDUAL_EPSILON / (1.0 - RESIDUAL_EPSILON))
    upper = -lower
    bounded = np.clip(np.asarray(scores, dtype=np.float64), lower, upper)
    return 1.0 / (1.0 + np.exp(-bounded))


def _logit(probability: float) -> float:
    probability = float(
        np.clip(probability, RESIDUAL_EPSILON, 1.0 - RESIDUAL_EPSILON)
    )
    return float(np.log(probability / (1.0 - probability)))


def _logit_working_response(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Newton working responses and binomial curvature weights."""

    probabilities = np.clip(
        _sigmoid(scores),
        RESIDUAL_EPSILON,
        1.0 - RESIDUAL_EPSILON,
    )
    weights = np.clip(
        probabilities * (1.0 - probabilities),
        MIN_LOGIT_WEIGHT,
        None,
    )
    responses = (np.asarray(labels, dtype=np.float64) - probabilities) / weights
    return responses, weights


def _load_ordered_events(
    dataset_path: str | PathLike[str],
    *,
    require_two_classes: bool = True,
) -> tuple[np.ndarray, ...]:
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
        ORDER BY event_timestamp, merchant_id
        """,
        params=[str(path), TRAIN_SPLIT],
    ).fetchnumpy()

    merchants = np.asarray(events[DATASET_MERCHANT_COLUMN])
    timestamps = np.asarray(events[DATASET_TIMESTAMP_COLUMN], dtype=np.int64)
    labels = np.asarray(events[DATASET_TARGET_COLUMN])
    if not len(labels):
        raise ValueError("Dataset has no rows in the training split")
    if require_two_classes and np.unique(labels).size < 2:
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

    def _values_for(self, individual: Any) -> np.ndarray:
        return self.materializer.materialize_for_events(
            individual,
            self.event_merchants,
            self.event_timestamps,
        )

    def prepare_population(self, individuals: Sequence[Any]) -> None:
        """Calculate and cache every feature before population evaluation."""

        for individual in individuals:
            self._values_for(individual)

    def __call__(self, individual: Any) -> float:
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
            logger.info(
                "Split %d/%d %s score=%.4f",
                len(self.fold_scores),
                len(self.cv_splits),
                individual,
                score,
            )

        self.last_model = self.last_models[-1]
        return float(np.mean(self.fold_scores))

    evaluate = __call__


class ResidualEvaluator:
    """Score a candidate by its out-of-sample improvement over a baseline.

    Each chronological fold starts with an intercept-only model in logit
    space. A shallow decision-tree regressor is fitted to the binomial Newton
    working response, weighted by the corresponding logit curvature. Its
    validation prediction is shrunk and added to the baseline logit before a
    sigmoid converts it back to a probability. The returned score is the
    relative validation Brier-score improvement, ``1 - corrected / baseline``.

    This is the residual proxy used by ``gp-benchmarks`` when its active set is
    empty. This evaluator remains stateless across candidates because the
    automatedfe search does not currently maintain an active-set archive.
    """

    def __init__(
        self,
        materializer: FeatureMaterializer,
        dataset_path: str | PathLike[str],
        *,
        n_splits: int = DEFAULT_N_SPLITS,
        score_metric: str = "brier_improvement",
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if score_metric not in {"brier", "brier_improvement"}:
            raise ValueError(
                "score_metric must be 'brier_improvement' or 'brier'"
            )
        self.materializer = materializer
        self.dataset_path = Path(dataset_path).resolve()
        self.n_splits = n_splits
        # ``brier`` is accepted as a short spelling, but the score is always
        # an improvement so that higher fitness remains better for GP.
        self.score_metric = "brier_improvement"
        (
            self.event_merchants,
            self.event_timestamps,
            self.labels,
        ) = _load_ordered_events(self.dataset_path, require_two_classes=False)
        unique_labels = np.unique(self.labels)
        if not np.all(np.isin(unique_labels, (0, 1))):
            raise ValueError("Residual evaluation requires binary target labels")
        if unique_labels.size < 2:
            raise ValueError("Residual evaluation requires at least two target classes")
        self.labels = self.labels.astype(np.float64, copy=False)
        self.cv_splits = _time_series_splits(self.event_timestamps, n_splits)

        # Keep the same convenient single-fold attributes as the existing
        # evaluator, while exposing all fold diagnostics for the residual path.
        self.fit_indices, self.validation_indices = self.cv_splits[-1]
        self.fold_scores: list[float] = []
        self.fold_baselines: list[float] = []
        self.fold_baseline_brier_scores: list[float] = []
        self.fold_corrected_brier_scores: list[float] = []
        self.last_models: list[DecisionTreeRegressor | None] = []
        self.last_model: DecisionTreeRegressor | None = None
        self.last_training_residuals: list[np.ndarray] = []
        self.last_training_weights: list[np.ndarray] = []
        self.last_validation_predictions: list[np.ndarray] = []

    def _values_for(self, individual: Any) -> np.ndarray:
        return self.materializer.materialize_for_events(
            individual,
            self.event_merchants,
            self.event_timestamps,
        )

    def prepare_population(self, individuals: Sequence[Any]) -> None:
        """Calculate and cache every candidate signal before evaluation."""

        for individual in individuals:
            self._values_for(individual)

    def __call__(self, individual: Any) -> float:
        values = self._values_for(individual).reshape(-1, 1)
        self.fold_scores = []
        self.fold_baselines = []
        self.fold_baseline_brier_scores = []
        self.fold_corrected_brier_scores = []
        self.last_models = []
        self.last_training_residuals = []
        self.last_training_weights = []
        self.last_validation_predictions = []

        for fit_indices, validation_indices in self.cv_splits:
            x_train = values[fit_indices]
            x_validation = values[validation_indices]
            y_train = self.labels[fit_indices]
            y_validation = self.labels[validation_indices]

            imputer = SimpleImputer(strategy="constant", fill_value=0.0)
            x_train = imputer.fit_transform(x_train)
            x_validation = imputer.transform(x_validation)

            baseline = float(
                np.clip(
                    np.mean(y_train),
                    RESIDUAL_EPSILON,
                    1.0 - RESIDUAL_EPSILON,
                )
            )
            baseline_score = _logit(baseline)
            training_scores = np.full(y_train.shape, baseline_score)
            validation_scores = np.full(y_validation.shape, baseline_score)
            baseline_predictions = _sigmoid(validation_scores)
            training_residuals, training_weights = _logit_working_response(
                y_train,
                training_scores,
            )

            model: DecisionTreeRegressor | None = None
            if np.std(x_train[:, 0]) < 1e-8:
                corrected_predictions = baseline_predictions.copy()
            else:
                model = DecisionTreeRegressor(**RESIDUAL_TREE_PARAMS)
                model.fit(
                    x_train,
                    training_residuals,
                    sample_weight=training_weights,
                )
                correction = model.predict(x_validation)
                corrected_predictions = _sigmoid(
                    validation_scores + RESIDUAL_SHRINKAGE * correction
                )
            baseline_brier = float(
                brier_score_loss(y_validation, baseline_predictions)
            )
            corrected_brier = float(
                brier_score_loss(y_validation, corrected_predictions)
            )
            improvement = (
                0.0
                if baseline_brier <= 1e-12
                else 1.0 - (corrected_brier / baseline_brier)
            )

            self.last_models.append(model)
            self.last_training_residuals.append(training_residuals.copy())
            self.last_training_weights.append(training_weights.copy())
            self.last_validation_predictions.append(corrected_predictions.copy())
            self.fold_baselines.append(baseline)
            self.fold_baseline_brier_scores.append(baseline_brier)
            self.fold_corrected_brier_scores.append(corrected_brier)
            self.fold_scores.append(float(improvement))
            logger.info(
                "Split %d/%d %s Brier improvement=%.6f",
                len(self.fold_scores),
                len(self.cv_splits),
                individual,
                improvement,
            )

        self.last_model = self.last_models[-1]
        return float(np.mean(self.fold_scores))

    evaluate = __call__


ResidualFitness = ResidualEvaluator
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
    "MIN_LOGIT_WEIGHT",
    "RESIDUAL_EPSILON",
    "RESIDUAL_SHRINKAGE",
    "RESIDUAL_TREE_PARAMS",
    "ResidualEvaluator",
    "ResidualFitness",
    "TRAIN_SPLIT",
]

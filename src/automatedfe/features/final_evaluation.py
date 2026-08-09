"""Final evaluation of a generated feature set on the held-out test split."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score

from .archive import ArchiveSnapshot, ArchiveStep, load_archive
from .feature_materialization import FeatureMaterializer
from .feature_types import TxFeature
from .fitness import (
    DATASET_MERCHANT_COLUMN,
    DATASET_TARGET_COLUMN,
    DATASET_TIMESTAMP_COLUMN,
    DEFAULT_RANDOM_STATE,
    TRAIN_SPLIT,
)
from .grammar import build_grammar

TEST_SPLIT = "test"
ARCHIVE_MINIMIZE = (False, False, False, True)
DEFAULT_N_ESTIMATORS = 50
DEFAULT_RF_MAX_DEPTH = 10
DEFAULT_RF_MIN_SAMPLES_LEAF = 2
DEFAULT_RF_MAX_SAMPLES = 100_000

logger = logging.getLogger(__name__)

ArchiveSource = ArchiveSnapshot | ArchiveStep | str | PathLike[str]


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


def _as_expression(individual: Any) -> Any:
    """Return a phenotype when *individual* is a Genetic Engine individual."""

    get_phenotype = getattr(individual, "get_phenotype", None)
    if callable(get_phenotype):
        return get_phenotype()
    return individual


def _deduplicate_individuals(individuals: Sequence[Any]) -> list[Any]:
    seen: set[str] = set()
    unique: list[Any] = []
    for individual in individuals:
        individual = _as_expression(individual)
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


def _validate_archive_snapshot(
    snapshot: ArchiveSnapshot,
    *,
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None,
) -> None:
    """Validate the archive configuration supported by final evaluation."""

    if not snapshot.expressions:
        raise ValueError("Archive is empty; at least one expression is required")
    if snapshot.minimize != ARCHIVE_MINIMIZE:
        raise ValueError(
            "Archive objective directions are incompatible with final evaluation; "
            f"expected {list(ARCHIVE_MINIMIZE)}, got {list(snapshot.minimize)}"
        )
    if len(snapshot.objectives) != len(snapshot.expressions):
        raise ValueError("Archive expressions and objectives have different lengths")

    if mapping is not None:
        # ``load_archive`` performs this check for paths. A snapshot may have
        # been loaded without a mapping, so apply the same validation here.
        from .archive import _resolve_mapping, _validate_mapping_compatible

        _validate_mapping_compatible(snapshot.mapping, _resolve_mapping(mapping))


def _resolve_archive(
    source: ArchiveSource,
    *,
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None,
) -> Sequence[Any]:
    """Resolve a live archive or JSON snapshot into ordered expressions."""

    if isinstance(source, ArchiveStep):
        if not source.archive:
            raise ValueError("Archive is empty; at least one expression is required")
        problem = source._problem
        if problem is not None and tuple(problem.minimize) != ARCHIVE_MINIMIZE:
            raise ValueError(
                "Archive objective directions are incompatible with final evaluation; "
                f"expected {list(ARCHIVE_MINIMIZE)}, got {list(problem.minimize)}"
            )
        return [_as_expression(individual) for individual in source.archive]

    if isinstance(source, ArchiveSnapshot):
        snapshot = source
    elif isinstance(source, (str, PathLike)):
        snapshot = load_archive(source, mapping=mapping)
    else:
        raise TypeError(
            "Archive source must be an ArchiveStep, ArchiveSnapshot, or JSON path"
        )

    _validate_archive_snapshot(snapshot, mapping=mapping)
    # Category nodes store indexes into the configured grammar code lists. A
    # loaded archive carries the authoritative mapping, so configure the
    # grammar before any categorical expression is materialized.
    build_grammar(snapshot.mapping)
    return snapshot.expressions


@dataclass(frozen=True, slots=True)
class FinalEvaluationResult:
    """Test-split metrics and the fitted final model."""

    metrics: dict[str, float]
    model: RandomForestClassifier
    predictions: np.ndarray


class FinalEvaluator:
    """Materialize a generated feature set and score it on the test split.

    The complete feature matrix is computed over the union of training and
    test events, then a random forest is fitted on the training rows and
    scored on the held-out test rows. ``evaluate`` accepts the historical
    sequence of expressions, a live :class:`ArchiveStep`, an
    :class:`ArchiveSnapshot`, or a path to an archive JSON file.
    """

    def __init__(
        self,
        materializer: FeatureMaterializer,
        dataset_path: str | PathLike[str],
        *,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth: int | None = DEFAULT_RF_MAX_DEPTH,
        min_samples_leaf: int = DEFAULT_RF_MIN_SAMPLES_LEAF,
        max_samples: int | float | None = DEFAULT_RF_MAX_SAMPLES,
        n_jobs: int = -1,
        mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
        archive: ArchiveSource | None = None,
    ) -> None:
        self.materializer = materializer
        self.dataset_path = Path(dataset_path).resolve()
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.mapping = mapping
        self.archive = archive

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

    def evaluate(
        self,
        individuals: Sequence[Any] | ArchiveSource | None = None,
        *,
        archive: ArchiveSource | None = None,
    ) -> FinalEvaluationResult:
        """Fit the final model on the training rows and score the test rows.

        ``individuals`` may be a normal expression sequence or an archive
        source. When an archive is configured on the evaluator, calling
        ``evaluate()`` uses it. The explicit ``archive`` keyword is a
        convenience equivalent to passing the archive positionally.
        """

        if individuals is not None and archive is not None:
            raise TypeError("Pass either individuals or archive, not both")
        source = archive if archive is not None else individuals
        if source is None:
            source = self.archive
        if source is None:
            raise ValueError("An expression sequence or archive is required")

        if isinstance(source, (ArchiveStep, ArchiveSnapshot, str, PathLike)):
            individuals = _resolve_archive(source, mapping=self.mapping)
        else:
            individuals = source

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

        forest_max_samples = self.max_samples
        if isinstance(forest_max_samples, int):
            forest_max_samples = min(forest_max_samples, len(x_train))
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_samples=forest_max_samples,
            class_weight="balanced_subsample",
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
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

    def evaluate_archive(self, archive: ArchiveSource) -> FinalEvaluationResult:
        """Evaluate every expression in a live or persisted archive."""

        return self.evaluate(archive)


__all__ = [
    "ARCHIVE_MINIMIZE",
    "ArchiveSource",
    "TEST_SPLIT",
    "FinalEvaluationResult",
    "FinalEvaluator",
]

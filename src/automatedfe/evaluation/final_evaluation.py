"""Final evaluation of a generated feature set on the held-out test split."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from time import monotonic
from typing import Any

import duckdb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from ..analysis.artifacts import feature_id
from ..search.archive import ArchiveSnapshot, ArchiveStep, load_archive
from ..features.feature_materialization import FeatureMaterializer
from .fitness import (
    DATASET_MERCHANT_COLUMN,
    DATASET_TARGET_COLUMN,
    DATASET_TIMESTAMP_COLUMN,
    DEFAULT_RANDOM_STATE,
    RESIDUAL_SHRINKAGE,
    RESIDUAL_TREE_PARAMS,
    TRAIN_SPLIT,
    logit,
    logit_working_response,
    sigmoid,
)
from ..features.grammar import build_grammar
from ..search.search import canonical_expression_key

TEST_SPLIT = "test"
ARCHIVE_MINIMIZE = (False, False, False, True)
DEFAULT_N_ESTIMATORS = 500
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
        key = canonical_expression_key(individual)
        if key in seen:
            continue
        seen.add(key)
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
        from ..search.archive import _resolve_mapping, _validate_mapping_compatible

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
class FinalEvaluationDiagnostics:
    """Model-free diagnostics from one final-archive forest evaluation."""

    feature_ids: tuple[str, ...]
    feature_labels: tuple[str, ...]
    search_fold_scores: tuple[tuple[float | None, ...], ...]
    metrics: dict[str, float]
    tree_importances: np.ndarray
    importance_means: np.ndarray
    importance_stds: np.ndarray
    spearman_correlations: np.ndarray
    correlation_training_row_count: int
    materialization_seconds: np.ndarray

    @property
    def feature_count(self) -> int:
        return len(self.feature_ids)

    @property
    def total_materialization_seconds(self) -> float:
        return float(np.sum(self.materialization_seconds))

    @property
    def per_tree_importances(self) -> np.ndarray:
        """Descriptive alias for the retained per-tree impurity matrix."""

        return self.tree_importances

    @property
    def feature_importances(self) -> np.ndarray:
        """Mean impurity importance for each final feature."""

        return self.importance_means

    @property
    def feature_importance_variation(self) -> np.ndarray:
        """Across-tree standard deviation of impurity importance."""

        return self.importance_stds

    @property
    def correlations(self) -> np.ndarray:
        return self.spearman_correlations

    @property
    def training_row_count(self) -> int:
        return self.correlation_training_row_count

    @property
    def materialization_durations(self) -> np.ndarray:
        return self.materialization_seconds


@dataclass(frozen=True, slots=True)
class FinalEvaluationResult:
    """Test-split metrics and the fitted final model."""

    metrics: dict[str, float]
    model: RandomForestClassifier
    predictions: np.ndarray
    diagnostics: FinalEvaluationDiagnostics | None = None

    def _diagnostics_value(self, name: str) -> Any:
        if self.diagnostics is None:
            return None
        return getattr(self.diagnostics, name)

    @property
    def feature_ids(self) -> tuple[str, ...] | None:
        return self._diagnostics_value("feature_ids")

    @property
    def feature_labels(self) -> tuple[str, ...] | None:
        return self._diagnostics_value("feature_labels")

    @property
    def search_fold_scores(self) -> tuple[tuple[float | None, ...], ...] | None:
        return self._diagnostics_value("search_fold_scores")

    @property
    def importance_means(self) -> np.ndarray | None:
        return self._diagnostics_value("importance_means")

    @property
    def importance_stds(self) -> np.ndarray | None:
        return self._diagnostics_value("importance_stds")

    @property
    def spearman_correlations(self) -> np.ndarray | None:
        return self._diagnostics_value("spearman_correlations")

    @property
    def correlation_training_row_count(self) -> int | None:
        return self._diagnostics_value("correlation_training_row_count")

    @property
    def materialization_seconds(self) -> np.ndarray | None:
        return self._diagnostics_value("materialization_seconds")

    @property
    def total_materialization_seconds(self) -> float | None:
        if self.diagnostics is None:
            return None
        return self.diagnostics.total_materialization_seconds


@dataclass(frozen=True, slots=True)
class AdditiveEvaluationResult:
    """Held-out metrics from the sequential active residual ensemble.

    ``train_predictions`` and ``test_predictions`` are probabilities produced
    by the additive model.  ``models`` contains the residual tree for each
    non-constant active expression, in promotion order.  Constant expressions
    do not need a tree and are therefore omitted from that tuple.
    """

    metrics: dict[str, float]
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    models: tuple[DecisionTreeRegressor, ...]
    expressions: tuple[Any, ...]

    @property
    def predictions(self) -> np.ndarray:
        """Return the test probabilities under the RF-compatible name."""

        return self.test_predictions

    @property
    def train_auc(self) -> float:
        """Return the additive ensemble's training ROC AUC."""

        return self.metrics["train_auc"]

    @property
    def test_auc(self) -> float:
        """Return the additive ensemble's held-out ROC AUC."""

        return self.metrics["test_auc"]

    @property
    def train_roc_auc(self) -> float:
        """Descriptive alias for :attr:`train_auc`."""

        return self.train_auc

    @property
    def test_roc_auc(self) -> float:
        """Descriptive alias for :attr:`test_auc`."""

        return self.test_auc


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

        self.train_event_merchants = train_merchants
        self.train_event_timestamps = train_timestamps
        self.test_event_merchants = test_merchants
        self.test_event_timestamps = test_timestamps

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

    def _materialize_feature_with_duration(
        self,
        individual: Any,
    ) -> tuple[np.ndarray, float]:
        """Materialize one feature on all evaluation events once.

        The materializer retains the original computation duration for event
        features, including durations restored from its durable cache.  The
        fallback clock is for compatible custom materializers that implement
        only the historical ``materialize_for_events`` method.
        """

        started = monotonic()
        values = np.asarray(
            self.materializer.materialize_for_events(
                individual,
                self.event_merchants,
                self.event_timestamps,
            ),
            dtype=np.float64,
        )
        duration = None
        get_duration = getattr(
            self.materializer,
            "event_materialization_duration",
            None,
        )
        if callable(get_duration):
            duration = get_duration(
                individual,
                self.event_merchants,
                self.event_timestamps,
            )
        if duration is None:
            duration = monotonic() - started
        duration = float(duration)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Feature materialization duration must be finite and non-negative")
        return values, duration

    @staticmethod
    def _spearman_matrix(values: np.ndarray) -> np.ndarray:
        """Return a Spearman matrix, preserving NaN for constant columns."""

        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Spearman input must be a two-dimensional matrix")
        rows, columns = values.shape
        if rows == 0:
            return np.full((columns, columns), np.nan, dtype=np.float64)

        ranks = np.empty_like(values, dtype=np.float64)
        for column in range(columns):
            order = np.argsort(values[:, column], kind="mergesort")
            sorted_values = values[order, column]
            rank_values = np.empty(rows, dtype=np.float64)
            start = 0
            while start < rows:
                end = start + 1
                while end < rows and sorted_values[end] == sorted_values[start]:
                    end += 1
                rank_values[order[start:end]] = (start + end - 1) / 2.0 + 1.0
                start = end
            ranks[:, column] = rank_values

        centered = ranks - np.mean(ranks, axis=0)
        norms = np.linalg.norm(centered, axis=0)
        denominator = norms[:, None] * norms[None, :]
        numerator = centered.T @ centered
        with np.errstate(divide="ignore", invalid="ignore"):
            correlations = numerator / denominator
        correlations[~np.isfinite(correlations)] = np.nan
        return correlations

    @staticmethod
    def _deduplicate_scores(
        individuals: Sequence[Any],
        scores: Sequence[Sequence[float]] | None,
    ) -> tuple[list[Any], tuple[tuple[float | None, ...], ...]]:
        seen: set[str] = set()
        unique: list[Any] = []
        unique_scores: list[tuple[float | None, ...]] = []
        for index, individual in enumerate(individuals):
            expression = _as_expression(individual)
            key = canonical_expression_key(expression)
            if key in seen:
                continue
            seen.add(key)
            unique.append(expression)
            score = scores[index] if scores is not None and index < len(scores) else ()
            values: list[float | None] = []
            for position in range(3):
                if position >= len(score):
                    values.append(None)
                    continue
                value = float(score[position])
                values.append(value if math.isfinite(value) else None)
            unique_scores.append(tuple(values))
        return unique, tuple(unique_scores)

    def _resolve_evaluation_source(
        self,
        source: Any,
        explicit_scores: Sequence[Sequence[float]] | None,
    ) -> tuple[list[Any], tuple[tuple[float | None, ...], ...]]:
        if isinstance(source, ArchiveStep):
            individuals = [_as_expression(individual) for individual in source.archive]
            scores: Sequence[Sequence[float]] | None = explicit_scores
            if scores is None:
                problem = getattr(source, "_problem", None)
                candidate_scores: list[tuple[float, ...]] = []
                if problem is not None:
                    for individual in source.archive:
                        get_fitness = getattr(individual, "get_fitness", None)
                        if not callable(get_fitness):
                            candidate_scores = []
                            break
                        fitness = get_fitness(problem)
                        candidate_scores.append(
                            tuple(float(value) for value in fitness.fitness_components)
                        )
                scores = candidate_scores or None
            return self._deduplicate_scores(individuals, scores)

        if isinstance(source, ArchiveSnapshot):
            _validate_archive_snapshot(source, mapping=self.mapping)
            build_grammar(source.mapping)
            return self._deduplicate_scores(
                source.expressions,
                explicit_scores if explicit_scores is not None else source.objectives,
            )

        if isinstance(source, (str, PathLike)):
            snapshot = load_archive(source, mapping=self.mapping)
            return self._resolve_evaluation_source(snapshot, explicit_scores)

        individuals = list(source)
        return self._deduplicate_scores(individuals, explicit_scores)

    def _resolve_individuals(self, source: Any) -> list[Any]:
        """Resolve and structurally deduplicate a final-evaluation input."""

        if isinstance(source, (ArchiveStep, ArchiveSnapshot, str, PathLike)):
            individuals = _resolve_archive(source, mapping=self.mapping)
        else:
            individuals = source
        return _deduplicate_individuals(individuals)

    def evaluate(
        self,
        individuals: Sequence[Any] | ArchiveSource | None = None,
        *,
        archive: ArchiveSource | None = None,
        search_fold_scores: Sequence[Sequence[float]] | None = None,
        include_diagnostics: bool = True,
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

        individuals, resolved_scores = self._resolve_evaluation_source(
            source,
            search_fold_scores,
        )
        if not individuals:
            raise ValueError("At least one individual is required for final evaluation")
        columns: list[np.ndarray] = []
        durations: list[float] = []
        for individual in individuals:
            values, duration = self._materialize_feature_with_duration(individual)
            if values.ndim != 1 or len(values) != len(self.event_merchants):
                raise ValueError(
                    "Materialized final feature must have one value per evaluation event"
                )
            columns.append(values)
            durations.append(duration)
        matrix = np.column_stack(columns)
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
            "roc_auc": float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])),
        }
        diagnostics = None
        if include_diagnostics:
            tree_importances = np.asarray(
                [tree.feature_importances_ for tree in model.estimators_],
                dtype=np.float64,
            )
            importance_means = np.mean(tree_importances, axis=0)
            importance_stds = np.std(tree_importances, axis=0)
            diagnostics = FinalEvaluationDiagnostics(
                feature_ids=tuple(feature_id(individual) for individual in individuals),
                feature_labels=tuple(str(individual) for individual in individuals),
                search_fold_scores=resolved_scores,
                metrics=dict(metrics),
                tree_importances=tree_importances,
                importance_means=importance_means,
                importance_stds=importance_stds,
                spearman_correlations=self._spearman_matrix(x_train),
                correlation_training_row_count=int(x_train.shape[0]),
                materialization_seconds=np.asarray(durations, dtype=np.float64),
            )
        return FinalEvaluationResult(
            metrics=metrics,
            model=model,
            predictions=predictions,
            diagnostics=diagnostics,
        )

    def evaluate_active_set(
        self,
        individuals: Sequence[Any] | ArchiveSource | None = None,
        *,
        archive: ArchiveSource | None = None,
    ) -> FinalEvaluationResult | None:
        """Evaluate promoted expressions with the standard final RF.

        Active expressions are already filtered during promotion. This method
        deliberately performs no additional correlation filtering and treats
        an empty active set as a valid, metric-less result. The positional
        ``evaluate`` method retains its historical error for an empty full
        archive.
        """

        if individuals is not None and archive is not None:
            raise TypeError("Pass either individuals or archive, not both")
        source = archive if archive is not None else individuals
        if source is None:
            source = self.archive
        if source is None:
            return None

        resolved = self._resolve_individuals(source)
        if not resolved:
            return None
        return self.evaluate(resolved)

    # Keep both spellings available to callers describing this as an active
    # RF evaluation rather than an active-set evaluation.
    evaluate_active_rf = evaluate_active_set
    evaluate_active = evaluate_active_set

    def evaluate_additive_ensemble(
        self,
        individuals: Sequence[Any] | ArchiveSource | None = None,
        *,
        archive: ArchiveSource | None = None,
    ) -> AdditiveEvaluationResult | None:
        """Fit and score the sequential additive active residual ensemble.

        The intercept is the training-label prior in logit space. Each active
        expression then gets one depth-one weighted residual tree, fitted only
        on the full training split. The same tree is applied to training and
        test rows before the expression's shrunk correction is added to the
        running logit. Test labels are used only for the final ROC AUC.
        """

        if individuals is not None and archive is not None:
            raise TypeError("Pass either individuals or archive, not both")
        source = archive if archive is not None else individuals
        if source is None:
            source = self.archive
        if source is None:
            return None

        expressions = self._resolve_individuals(source)
        if not expressions:
            return None

        matrix = self._feature_matrix(expressions)
        y_train = np.asarray(self.labels[self.train_indices], dtype=np.float64)
        y_test = np.asarray(self.labels[self.test_indices])
        intercept_score = logit(float(np.mean(y_train)))
        train_scores = np.full(y_train.shape, intercept_score)
        test_scores = np.full(len(self.test_indices), intercept_score)
        models: list[DecisionTreeRegressor] = []

        for column_index, _expression in enumerate(expressions):
            # Fit the imputer on training rows only. The current constant
            # strategy has no learned statistic, but keeping the split boundary
            # explicit prevents a future strategy from leaking test data.
            imputer = SimpleImputer(strategy="constant", fill_value=0.0)
            train_values = imputer.fit_transform(
                matrix[self.train_indices, column_index].reshape(-1, 1)
            )
            test_values = imputer.transform(
                matrix[self.test_indices, column_index].reshape(-1, 1)
            )

            if np.std(train_values[:, 0]) < 1e-8:
                continue

            residuals, weights = logit_working_response(y_train, train_scores)
            model = DecisionTreeRegressor(**RESIDUAL_TREE_PARAMS)
            model.fit(train_values, residuals, sample_weight=weights)
            train_scores += RESIDUAL_SHRINKAGE * model.predict(train_values)
            test_scores += RESIDUAL_SHRINKAGE * model.predict(test_values)
            models.append(model)

        train_predictions = sigmoid(train_scores)
        test_predictions = sigmoid(test_scores)
        if np.unique(y_test).size < 2:
            raise ValueError("ROC AUC requires both target classes in the test split")

        return AdditiveEvaluationResult(
            metrics={
                "train_auc": float(roc_auc_score(y_train, train_predictions)),
                "test_auc": float(roc_auc_score(y_test, test_predictions)),
            },
            train_predictions=train_predictions,
            test_predictions=test_predictions,
            models=tuple(models),
            expressions=tuple(expressions),
        )

    # Short aliases make the additive path easy to discover without changing
    # the existing ``evaluate``/``evaluate_archive`` API.
    evaluate_active_additive = evaluate_additive_ensemble
    evaluate_additive = evaluate_additive_ensemble

    __call__ = evaluate

    def evaluate_archive(self, archive: ArchiveSource) -> FinalEvaluationResult:
        """Evaluate every expression in a live or persisted archive."""

        return self.evaluate(archive)


__all__ = [
    "ARCHIVE_MINIMIZE",
    "AdditiveEvaluationResult",
    "ArchiveSource",
    "TEST_SPLIT",
    "FinalEvaluationResult",
    "FinalEvaluationDiagnostics",
    "FinalEvaluator",
]

"""Unified orchestration for feature-search strategies.

The lower-level search builders intentionally remain available for callers
that need to customize Genetic Engine's lifecycle. This module provides the
common workflow used by experiments: configure one strategy, search, and
evaluate the resulting feature set on the held-out split.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from os import PathLike
from time import monotonic_ns
from typing import Any

from geneticengine.evaluation.budget import TimeBudget

from ..transaction_materialization import DEFAULT_MMAP_DIR
from .feature_materialization import FeatureMaterializer
from .final_evaluation import FinalEvaluationResult, FinalEvaluator
from .fitness import DEFAULT_N_SPLITS, DEFAULT_RANDOM_STATE
from .grammar import expr
from .search.enumerative_search import build_enumerative_search
from .search.gp import build_search_algorithm
from .search.random_search import build_random_search
from .search.unbound_enumerative_search import build_unbound_enumerative_search


class SearchStrategy(str, Enum):
    """Strategies supported by :func:`run_feature_search`."""

    GENETIC = "genetic"
    ENUMERATIVE = "enumerative"
    RANDOM = "random"
    ENUMERATIVE_WITHOUT_ARCHIVE = "enumerative_without_archive"

    def __str__(self) -> str:
        return self.value


_EVALUATED_STRATEGIES = frozenset(
    {
        SearchStrategy.GENETIC,
        SearchStrategy.ENUMERATIVE,
        SearchStrategy.RANDOM,
    }
)


@dataclass(frozen=True, slots=True)
class SearchRunResult:
    """Search diagnostics, selected expressions, and held-out evaluation."""

    strategy: SearchStrategy
    expressions: tuple[expr, ...]
    final_evaluation: FinalEvaluationResult
    search_duration_seconds: float
    final_evaluation_duration_seconds: float
    generated_count: int
    evaluated_count: int
    invalid_count: int
    duplicate_count: int
    objectives: tuple[tuple[float, ...], ...] | None
    grammar_exhausted: bool

    @property
    def final_metrics(self) -> dict[str, float]:
        """Return the held-out metrics without unpacking the final result."""

        return self.final_evaluation.metrics

    @property
    def final_result(self) -> FinalEvaluationResult:
        """Compatibility alias for the held-out evaluation result."""

        return self.final_evaluation

    @property
    def final_evaluation_result(self) -> FinalEvaluationResult:
        """Return the held-out result under its explicit type name."""

        return self.final_evaluation

    @property
    def metrics(self) -> dict[str, float]:
        """Compatibility alias for :attr:`final_metrics`."""

        return self.final_metrics

    @property
    def selected_expressions(self) -> tuple[expr, ...]:
        """Return the ordered expressions passed to final evaluation."""

        return self.expressions

    @property
    def archive_expressions(self) -> tuple[expr, ...]:
        """Return the selected archive or generated expression sequence."""

        return self.expressions

    @property
    def evaluation_count(self) -> int:
        """Compatibility alias for the number of search evaluations."""

        return self.evaluated_count

    @property
    def accepted_count(self) -> int:
        """Return generated candidates that were not structural duplicates."""

        return self.generated_count - self.duplicate_count

    @property
    def search_duration(self) -> float:
        """Compatibility alias for the search duration in seconds."""

        return self.search_duration_seconds

    @property
    def final_evaluation_duration(self) -> float:
        """Compatibility alias for the final-evaluation duration."""

        return self.final_evaluation_duration_seconds


def _coerce_strategy(strategy: SearchStrategy | str) -> SearchStrategy:
    if isinstance(strategy, SearchStrategy):
        return strategy
    try:
        return SearchStrategy(strategy)
    except (TypeError, ValueError) as error:
        values = ", ".join(member.value for member in SearchStrategy)
        raise ValueError(
            f"Unknown search strategy {strategy!r}; expected one of: {values}"
        ) from error


def _validate_time_budget(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("time_budget_seconds must be a finite positive number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError("time_budget_seconds must be a finite positive number")
    return converted


def _validate_candidate_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("candidate_count must be a positive integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError("candidate_count must be a positive integer")
    return converted


def  _validate_budget_contract(
    strategy: SearchStrategy,
    *,
    time_budget_seconds: object,
    candidate_count: object,
) -> tuple[float | None, int | None]:
    evaluated = strategy in _EVALUATED_STRATEGIES
    if evaluated:
        if time_budget_seconds is None:
            raise ValueError(
                f"strategy {strategy.value!r} requires a positive "
                "time_budget_seconds"
            )
        if candidate_count is not None:
            raise ValueError(
                f"strategy {strategy.value!r} does not accept candidate_count"
            )
        return _validate_time_budget(time_budget_seconds), None

    if candidate_count is None:
        raise ValueError(
            f"strategy {strategy.value!r} requires a positive candidate_count"
        )
    if time_budget_seconds is not None:
        raise ValueError(
            f"strategy {strategy.value!r} does not accept time_budget_seconds"
        )
    return None, _validate_candidate_count(candidate_count)


def _as_expression(individual: Any) -> expr:
    get_phenotype = getattr(individual, "get_phenotype", None)
    if callable(get_phenotype):
        individual = get_phenotype()
    return individual


def _archive_expressions_and_objectives(
    algorithm: Any,
    result: Sequence[Any] | None,
) -> tuple[tuple[expr, ...], tuple[tuple[float, ...], ...] | None]:
    """Extract the final archive in the order returned by the search."""

    individuals = list(result) if result is not None else []
    if not individuals:
        archive = getattr(algorithm, "archive", None)
        if archive is not None:
            individuals = list(getattr(archive, "archive", archive))

    expressions = tuple(_as_expression(individual) for individual in individuals)
    objectives: list[tuple[float, ...]] = []
    problem = getattr(algorithm, "problem", None)
    for individual in individuals:
        get_fitness = getattr(individual, "get_fitness", None)
        if problem is None or not callable(get_fitness):
            # A custom search implementation may return expressions without
            # exposing Genetic Engine fitness objects. The expressions are
            # still useful, but objective diagnostics are unavailable.
            return expressions, None
        fitness = get_fitness(problem)
        objectives.append(tuple(float(value) for value in fitness.fitness_components))
    return expressions, tuple(objectives)


def _reset_search_clock(algorithm: Any, started_ns: int) -> None:
    """Make a Genetic Engine time budget begin after one-time setup."""

    tracker = getattr(algorithm, "tracker", None)
    if tracker is not None and hasattr(tracker, "start_time"):
        tracker.start_time = started_ns


def _search_counts(search: Any) -> tuple[int, int, int, int]:
    tracker = getattr(search, "tracker", None)
    evaluated = 0
    get_number_evaluations = getattr(tracker, "get_number_evaluations", None)
    if callable(get_number_evaluations):
        evaluated = int(get_number_evaluations())
    return (
        int(getattr(search, "generated_count", evaluated)),
        evaluated,
        int(getattr(search, "invalid_count", 0)),
        int(getattr(search, "duplicate_count", 0)),
    )


def _build_final_evaluator(
    materializer: FeatureMaterializer,
    dataset_path: str | PathLike[str],
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None,
) -> FinalEvaluator:
    return FinalEvaluator(materializer, dataset_path, mapping=mapping)


def _empty_archive_error(
    strategy: SearchStrategy,
    *,
    generated_count: int,
    evaluated_count: int,
    invalid_count: int,
    duplicate_count: int,
    grammar_exhausted: bool,
    search_duration_seconds: float,
) -> ValueError:
    return ValueError(
        "Evaluated feature search produced an empty archive: "
        f"strategy={strategy.value!r}, generated={generated_count}, "
        f"evaluated={evaluated_count}, invalid={invalid_count}, "
        f"duplicates={duplicate_count}, grammar_exhausted={grammar_exhausted}, "
        f"search_duration_seconds={search_duration_seconds:.6f}"
    )


def run_feature_search(
    strategy: SearchStrategy | str,
    *,
    time_budget_seconds: float | None = None,
    candidate_count: int | None = None,
    dataset_path: str | PathLike[str],
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    mmap_dir: str | PathLike[str] = DEFAULT_MMAP_DIR,
    feature_cache_dir: str | PathLike[str] | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    score_metric: str = "roc_auc",
    fitness_random_state: int = DEFAULT_RANDOM_STATE,
    seed: int = 42,
    population_size: int = 20,
    max_depth: int | None = None,
) -> SearchRunResult:
    """Run one strategy and always evaluate its selected features on test data.

    ``genetic``, ``enumerative``, and ``random`` use a positive wall-clock
    search budget. ``enumerative_without_archive`` uses a positive number of
    generated expressions instead and performs no search-time materialization
    or fitness evaluation. Search setup is completed before the search timer
    starts; held-out evaluation is timed separately.
    """

    selected_strategy = _coerce_strategy(strategy)
    validated_time_budget, validated_candidate_count = _validate_budget_contract(
        selected_strategy,
        time_budget_seconds=time_budget_seconds,
        candidate_count=candidate_count,
    )

    # Building the search object and its materializer is setup. In particular,
    # constructing an evaluated builder can load the dataset and feature
    # cache, so the runner starts its search clock only below.
    search: Any
    materializer: FeatureMaterializer
    if selected_strategy is SearchStrategy.GENETIC:
        search = build_search_algorithm(
            TimeBudget(validated_time_budget),
            mapping=mapping,
            population_size=population_size,
            seed=seed,
            mmap_dir=mmap_dir,
            feature_cache_dir=feature_cache_dir,
            dataset_path=dataset_path,
            n_splits=n_splits,
            score_metric=score_metric,
            fitness_random_state=fitness_random_state,
            max_depth=max_depth,
        )
        materializer = search.materializer
    elif selected_strategy is SearchStrategy.ENUMERATIVE:
        search = build_enumerative_search(
            TimeBudget(validated_time_budget),
            mapping=mapping,
            seed=seed,
            mmap_dir=mmap_dir,
            feature_cache_dir=feature_cache_dir,
            dataset_path=dataset_path,
            n_splits=n_splits,
            score_metric=score_metric,
            fitness_random_state=fitness_random_state,
            max_depth=max_depth,
        )
        materializer = search.materializer
    elif selected_strategy is SearchStrategy.RANDOM:
        search = build_random_search(
            TimeBudget(validated_time_budget),
            mapping=mapping,
            seed=seed,
            mmap_dir=mmap_dir,
            feature_cache_dir=feature_cache_dir,
            dataset_path=dataset_path,
            n_splits=n_splits,
            score_metric=score_metric,
            fitness_random_state=fitness_random_state,
            max_depth=max_depth,
        )
        materializer = search.materializer
    else:
        search = build_unbound_enumerative_search(
            validated_candidate_count,
            mapping=mapping,
            max_depth=max_depth,
        )
        materializer = FeatureMaterializer(
            mmap_dir,
            features_dir=feature_cache_dir,
        )

    search_started_ns = monotonic_ns()
    if selected_strategy in _EVALUATED_STRATEGIES:
        _reset_search_clock(search, search_started_ns)
    search_output = search.search()
    search_duration_seconds = (monotonic_ns() - search_started_ns) * 1e-9

    if selected_strategy in _EVALUATED_STRATEGIES:
        expressions, objectives = _archive_expressions_and_objectives(
            search, search_output
        )
    else:
        expressions = tuple(_as_expression(expression) for expression in search_output)
        objectives = None

    generated_count, evaluated_count, invalid_count, duplicate_count = _search_counts(
        search
    )
    grammar_exhausted = bool(getattr(search, "grammar_exhausted", False))

    if selected_strategy in _EVALUATED_STRATEGIES and not expressions:
        raise _empty_archive_error(
            selected_strategy,
            generated_count=generated_count,
            evaluated_count=evaluated_count,
            invalid_count=invalid_count,
            duplicate_count=duplicate_count,
            grammar_exhausted=grammar_exhausted,
            search_duration_seconds=search_duration_seconds,
        )

    final_started_ns = monotonic_ns()
    final_evaluator = _build_final_evaluator(materializer, dataset_path, mapping)
    final_evaluation = final_evaluator.evaluate(expressions)
    final_evaluation_duration_seconds = (monotonic_ns() - final_started_ns) * 1e-9

    return SearchRunResult(
        strategy=selected_strategy,
        expressions=expressions,
        final_evaluation=final_evaluation,
        search_duration_seconds=search_duration_seconds,
        final_evaluation_duration_seconds=final_evaluation_duration_seconds,
        generated_count=generated_count,
        evaluated_count=evaluated_count,
        invalid_count=invalid_count,
        duplicate_count=duplicate_count,
        objectives=objectives,
        grammar_exhausted=grammar_exhausted,
    )


__all__ = ["SearchRunResult", "SearchStrategy", "run_feature_search"]

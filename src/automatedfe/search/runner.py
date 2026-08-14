"""Unified orchestration for feature-search strategies.

The lower-level search builders intentionally remain available for callers
that need to customize Genetic Engine's lifecycle. This module provides the
common workflow used by experiments: configure one strategy, search, and
evaluate the resulting feature set on the held-out split.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Integral, Real
from os import PathLike
from pathlib import Path
from time import monotonic_ns
from typing import TYPE_CHECKING, Any

from geneticengine.evaluation.budget import TimeBudget

from ..analysis.run_report import render_run_report
from ..data.transaction_materialization import DEFAULT_MMAP_DIR
from ..evaluation.final_evaluation import (
    AdditiveEvaluationResult,
    FinalEvaluationResult,
    FinalEvaluator,
)
from ..evaluation.fitness import DEFAULT_N_SPLITS, DEFAULT_RANDOM_STATE
from ..features.feature_materialization import FeatureMaterializer
from ..features.grammar import expr
from ..tracking import MlflowRunStore
from .enumerative_search import build_enumerative_search
from .gp import build_search_algorithm
from .lifecycle import SearchLifecycleRecorder
from .random_search import build_random_search
from .search import canonical_expression_key
from .unbound_enumerative_search import build_unbound_enumerative_search

if TYPE_CHECKING:
    from ..analysis.run_bundle import RunBundleWriter


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
    """Search diagnostics, selected expressions, and held-out evaluations."""

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
    active_set_expressions: tuple[expr, ...] = ()
    active_set_final_evaluation: FinalEvaluationResult | None = None
    active_set_final_evaluation_duration_seconds: float | None = None
    additive_evaluation: AdditiveEvaluationResult | None = None
    additive_evaluation_duration_seconds: float | None = None
    history_count: int = 0
    active_set_count: int = 0
    lifecycle: SearchLifecycleRecorder | None = None
    run_id: str | None = None

    @property
    def final_metrics(self) -> dict[str, float]:
        """Return the held-out metrics without unpacking the final result."""

        return self.final_evaluation.metrics

    @property
    def final_evaluation_result(self) -> FinalEvaluationResult:
        """Return the held-out result under its explicit type name."""

        return self.final_evaluation

    @property
    def selected_expressions(self) -> tuple[expr, ...]:
        """Return the ordered expressions passed to final evaluation."""

        return self.expressions

    @property
    def archive_expressions(self) -> tuple[expr, ...]:
        """Return the selected archive or generated expression sequence."""

        return self.expressions

    @property
    def archive_final_evaluation(self) -> FinalEvaluationResult:
        """Return the primary held-out evaluation of the archive features."""

        return self.final_evaluation

    @property
    def active_set_final_metrics(self) -> dict[str, float] | None:
        """Return held-out metrics for promoted active features, when present."""

        if self.active_set_final_evaluation is None:
            return None
        return self.active_set_final_evaluation.metrics

    @property
    def additive_metrics(self) -> dict[str, float] | None:
        """Return train/test ROC AUC from the active additive ensemble."""

        if self.additive_evaluation is None:
            return None
        return self.additive_evaluation.metrics

    @property
    def accepted_count(self) -> int:
        """Return generated candidates that were not structural duplicates."""

        return self.generated_count - self.duplicate_count

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


def _validate_budget_contract(
    strategy: SearchStrategy,
    *,
    time_budget_seconds: object,
    candidate_count: object,
) -> tuple[float | None, int | None]:
    evaluated = strategy in _EVALUATED_STRATEGIES
    if evaluated:
        if time_budget_seconds is None:
            raise ValueError(
                f"strategy {strategy.value!r} requires a positive time_budget_seconds"
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


def _preflight_output_paths(
    strategy: SearchStrategy,
    *,
    csv_path: str | PathLike[str] | None,
    archive_path: str | PathLike[str] | None,
    history_path: str | PathLike[str] | None,
    active_archive_path: str | PathLike[str] | None,
    use_active_set: bool,
    force: bool,
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    """Validate all runner outputs before setup or file creation begins."""

    if not isinstance(force, bool):
        raise ValueError("force must be a boolean")
    if not isinstance(use_active_set, bool):
        raise ValueError("use_active_set must be a boolean")
    if (
        strategy is SearchStrategy.ENUMERATIVE_WITHOUT_ARCHIVE
        and archive_path is not None
    ):
        raise ValueError(
            "archive_path is not supported for enumerative_without_archive"
        )
    if not use_active_set and (
        history_path is not None or active_archive_path is not None
    ):
        raise ValueError(
            "history_path and active_archive_path require use_active_set=True"
        )

    resolved_csv = Path(csv_path).resolve() if csv_path is not None else None
    resolved_archive = (
        Path(archive_path).resolve() if archive_path is not None else None
    )
    resolved_history = (
        Path(history_path).resolve() if history_path is not None else None
    )
    resolved_active_archive = (
        Path(active_archive_path).resolve() if active_archive_path is not None else None
    )
    outputs = [
        path
        for path in (
            resolved_csv,
            resolved_archive,
            resolved_history,
            resolved_active_archive,
        )
        if path is not None
    ]
    if len(set(outputs)) != len(outputs):
        raise ValueError(
            "csv_path, archive_path, history_path, and active_archive_path "
            "must identify different files"
        )
    for path in outputs:
        if path.exists() and path.is_dir():
            raise ValueError(
                f"Output path must identify a file, not a directory: {path}"
            )
        if path.exists() and not force:
            raise FileExistsError(
                f"Refusing to overwrite existing output without force=True: {path}"
            )
    return resolved_csv, resolved_archive, resolved_history, resolved_active_archive


def _as_expression(individual: Any) -> expr:
    get_phenotype = getattr(individual, "get_phenotype", None)
    if callable(get_phenotype):
        individual = get_phenotype()
    return individual


def _archive_expressions_and_objectives(
    algorithm: Any,
    result: Sequence[Any] | None,
) -> tuple[tuple[expr, ...], tuple[tuple[float, ...], ...] | None]:
    """Extract the complete permanent archive in stable admission order."""

    archive_owner = getattr(algorithm, "archive", None)
    archive_members = getattr(archive_owner, "archive", None)
    if archive_members is not None:
        individuals = list(archive_members)
    else:
        individuals = list(result) if result is not None else []

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
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> FinalEvaluator:
    return FinalEvaluator(
        materializer,
        dataset_path,
        mapping=mapping,
        random_state=random_state,
    )


def _evaluate_final_archive(
    evaluator: FinalEvaluator,
    expressions: Sequence[expr],
    objectives: tuple[tuple[float, ...], ...] | None,
) -> Any:
    """Evaluate the archive once, attaching its search-fold diagnostics."""

    return evaluator.evaluate(
        expressions,
        search_fold_scores=objectives,
    )


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


def _run_feature_search_impl(
    strategy: SearchStrategy | str,
    *,
    time_budget_seconds: float | None = None,
    candidate_count: int | None = None,
    dataset_path: str | PathLike[str],
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    mmap_dir: str | PathLike[str] = DEFAULT_MMAP_DIR,
    feature_cache_dir: str | PathLike[str] | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    score_metric: str = "brier_improvement",
    fitness_random_state: int = DEFAULT_RANDOM_STATE,
    seed: int = 42,
    population_size: int = 50,
    max_depth: int | None = None,
    use_active_set: bool = False,
    promotion_interval: int = 5,
    first_promotion_top_k: int = 2,
    promotion_add_k: int = 1,
    promotion_refresh_top_n: int = 50,
    archive_quality_threshold: float = 0.001,
    archive_correlation_threshold: float = 0.85,
    active_correlation_threshold: float = 0.90,
    promotion_min_gain: float = 0.0,
    promotion_mean_gain: float = 0.0005,
    csv_path: str | PathLike[str] | None = None,
    archive_path: str | PathLike[str] | None = None,
    history_path: str | PathLike[str] | None = None,
    active_archive_path: str | PathLike[str] | None = None,
    force: bool = False,
    _bundle_writer: RunBundleWriter | None = None,
) -> SearchRunResult:
    """Run one strategy and always evaluate its selected features on test data.

    ``genetic``, ``enumerative``, and ``random`` use a positive wall-clock
    search budget. ``enumerative_without_archive`` uses a positive number of
    generated expressions instead and performs no search-time materialization
    or fitness evaluation. Search setup is completed before the search timer
    starts; held-out evaluation is timed separately. ``csv_path`` records the
    common incremental diagnostics, while ``archive_path`` saves an evaluated
    strategy's complete permanent archive once. In genetic active-set mode, the
    Pareto archive and promoted active set are fitted and scored separately on
    the held-out split, ``history_path`` persists the complete filtered
    history, and ``active_archive_path`` persists the versioned active
    snapshot. Existing outputs require ``force=True``.
    """

    selected_strategy = _coerce_strategy(strategy)
    if not isinstance(use_active_set, bool):
        raise ValueError("use_active_set must be a boolean")
    if use_active_set and selected_strategy is not SearchStrategy.GENETIC:
        raise ValueError("use_active_set is supported only by the genetic strategy")
    if use_active_set and score_metric != "brier_improvement":
        raise ValueError("use_active_set requires score_metric='brier_improvement'")
    validated_time_budget, validated_candidate_count = _validate_budget_contract(
        selected_strategy,
        time_budget_seconds=time_budget_seconds,
        candidate_count=candidate_count,
    )
    (
        resolved_csv_path,
        resolved_archive_path,
        resolved_history_path,
        resolved_active_archive_path,
    ) = _preflight_output_paths(
        selected_strategy,
        csv_path=csv_path,
        archive_path=archive_path,
        history_path=history_path,
        active_archive_path=active_archive_path,
        use_active_set=use_active_set,
        force=force,
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
            use_active_set=use_active_set,
            promotion_interval=promotion_interval,
            first_promotion_top_k=first_promotion_top_k,
            promotion_add_k=promotion_add_k,
            promotion_refresh_top_n=promotion_refresh_top_n,
            archive_quality_threshold=archive_quality_threshold,
            archive_correlation_threshold=archive_correlation_threshold,
            active_correlation_threshold=active_correlation_threshold,
            promotion_min_gain=promotion_min_gain,
            promotion_mean_gain=promotion_mean_gain,
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

    lifecycle: SearchLifecycleRecorder | None = None
    search_started_ns = monotonic_ns()
    try:
        lifecycle = SearchLifecycleRecorder(
            strategy=selected_strategy.value,
            candidate_csv_path=(
                str(_bundle_writer.staged_candidates_path)
                if _bundle_writer is not None
                else (str(resolved_csv_path) if resolved_csv_path is not None else None)
            ),
        )
        if _bundle_writer is not None:
            _bundle_writer.lifecycle = lifecycle
        if selected_strategy in _EVALUATED_STRATEGIES:
            search.lifecycle = lifecycle
        else:
            search.candidate_observers.append(lifecycle.record_generated)
        if selected_strategy in _EVALUATED_STRATEGIES:
            _reset_search_clock(search, search_started_ns)
        search_output = search.search()
    except BaseException:
        if lifecycle is not None:
            lifecycle.close()
        raise
    search_duration_seconds = (monotonic_ns() - search_started_ns) * 1e-9

    active_set_expressions: tuple[expr, ...] = ()
    active_mode = bool(
        selected_strategy is SearchStrategy.GENETIC
        and (
            getattr(search, "active_set_manager", None) is not None
            or getattr(getattr(search, "archive_step", None), "use_active_set", False)
        )
    )
    if active_mode:
        expressions, objectives = _archive_expressions_and_objectives(
            search,
            getattr(search.archive_step, "archive", ()),
        )
        active_set_expressions = tuple(
            _as_expression(individual)
            for individual in getattr(
                getattr(search, "active_set_manager", search.archive_step),
                "active_individuals",
                (),
            )
        )
    elif selected_strategy in _EVALUATED_STRATEGIES:
        expressions, objectives = _archive_expressions_and_objectives(
            search, search_output
        )
    else:
        expressions = tuple(_as_expression(expression) for expression in search_output)
        objectives = None

    generated_count, evaluated_count, invalid_count, duplicate_count = _search_counts(
        search
    )
    grammar_exhausted = (
        search.grammar_exhausted
        if selected_strategy in _EVALUATED_STRATEGIES
        else search.exhausted
    )

    if lifecycle is not None:
        lifecycle.on_search_completed(
            canonical_expression_key(expression)
            for expression in (
                expressions if selected_strategy in _EVALUATED_STRATEGIES else ()
            )
        )

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

    if resolved_archive_path is not None:
        search.archive_step.save(resolved_archive_path, mapping=mapping)

    if resolved_history_path is not None or resolved_active_archive_path is not None:
        manager = getattr(search, "active_set_manager", None)
        if manager is None:
            raise TypeError(
                "history_path and active_archive_path require an active-set manager"
            )
        if resolved_history_path is not None:
            manager.save_history(resolved_history_path, mapping=mapping)
        if resolved_active_archive_path is not None:
            manager.save_active_snapshot(resolved_active_archive_path, mapping=mapping)

    final_started_ns = monotonic_ns()
    final_evaluator = _build_final_evaluator(
        materializer,
        dataset_path,
        mapping,
        random_state=seed,
    )
    final_evaluation = _evaluate_final_archive(
        final_evaluator,
        expressions,
        objectives,
    )
    final_evaluation_duration_seconds = (monotonic_ns() - final_started_ns) * 1e-9

    if (
        _bundle_writer is not None
        and isinstance(final_evaluation, FinalEvaluationResult)
        and final_evaluation.diagnostics is not None
    ):
        _bundle_writer.write_evaluation(
            final_evaluation,
            search_fold_metric=(
                "brier_improvement"
                if score_metric in {"brier", "brier_improvement"}
                else score_metric
            ),
        )

    active_set_final_evaluation: FinalEvaluationResult | None = None
    active_set_final_evaluation_duration_seconds: float | None = None
    additive_evaluation: AdditiveEvaluationResult | None = None
    additive_evaluation_duration_seconds: float | None = None
    if active_mode and active_set_expressions:
        active_started_ns = monotonic_ns()
        active_set_final_evaluation = final_evaluator.evaluate(
            active_set_expressions,
            include_diagnostics=False,
        )
        active_set_final_evaluation_duration_seconds = (
            monotonic_ns() - active_started_ns
        ) * 1e-9
        additive_started_ns = monotonic_ns()
        additive_evaluation = final_evaluator.evaluate_additive_ensemble(
            active_set_expressions
        )
        additive_evaluation_duration_seconds = (
            monotonic_ns() - additive_started_ns
        ) * 1e-9

    active_count_source = getattr(search, "active_set_manager", None)
    if active_count_source is None:
        active_count_source = getattr(search, "archive_step", None)
    history_count = len(getattr(active_count_source, "history_individuals", ()))

    result = SearchRunResult(
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
        active_set_expressions=active_set_expressions,
        active_set_final_evaluation=active_set_final_evaluation,
        active_set_final_evaluation_duration_seconds=(
            active_set_final_evaluation_duration_seconds
        ),
        additive_evaluation=additive_evaluation,
        additive_evaluation_duration_seconds=additive_evaluation_duration_seconds,
        history_count=history_count,
        active_set_count=len(active_set_expressions),
        lifecycle=lifecycle,
    )
    if lifecycle is not None:
        lifecycle.close()
    return result


class SearchAnalysisError(RuntimeError):
    """Automatic analysis failed after the search bundle was completed."""

    def __init__(self, run_id: str, error: BaseException) -> None:
        super().__init__(f"Automatic analysis failed for MLflow run {run_id}: {error}")
        self.run_id = run_id


_STRATEGY_GROUPS = {
    SearchStrategy.ENUMERATIVE: "archive_filtered_enumeration",
    SearchStrategy.RANDOM: "random_search",
    SearchStrategy.ENUMERATIVE_WITHOUT_ARCHIVE: "unfiltered_enumeration_benchmark",
}

_GENERATION_METRICS = {
    "Generated": "generated",
    "Unique": "unique",
    "Duplicate": "duplicate",
    "Invalid": "invalid",
    "Evaluated": "evaluated",
    "ArchiveSize": "archive_size",
    "Added": "added",
    "DurationSeconds": "generation_duration_seconds",
    "CumulativeRuntimeSeconds": "cumulative_runtime_seconds",
}


def _strategy_group(strategy: SearchStrategy, *, use_active_set: bool) -> str:
    if strategy is SearchStrategy.GENETIC:
        return "cost_effective_gp" if use_active_set else "standard_gp"
    return _STRATEGY_GROUPS[strategy]


def _path_parameter(value: str | PathLike[str] | None) -> str | None:
    return None if value is None else str(Path(value).expanduser().resolve())


def _run_parameters(
    *,
    time_budget_seconds: float | None,
    candidate_count: int | None,
    dataset_path: str | PathLike[str],
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None,
    mmap_dir: str | PathLike[str],
    feature_cache_dir: str | PathLike[str] | None,
    n_splits: int,
    score_metric: str,
    fitness_random_state: int,
    seed: int,
    population_size: int,
    max_depth: int | None,
    use_active_set: bool,
    promotion_interval: int,
    first_promotion_top_k: int,
    promotion_add_k: int,
    promotion_refresh_top_n: int,
    archive_quality_threshold: float,
    archive_correlation_threshold: float,
    active_correlation_threshold: float,
    promotion_min_gain: float,
    promotion_mean_gain: float,
    feature_labels: str,
) -> dict[str, object]:
    return {
        "time_budget_seconds": time_budget_seconds,
        "candidate_count": candidate_count,
        "dataset_path": _path_parameter(dataset_path),
        "mapping": (
            _path_parameter(mapping)
            if isinstance(mapping, (str, PathLike))
            else "inline"
        ),
        "mmap_dir": _path_parameter(mmap_dir),
        "feature_cache_dir": _path_parameter(feature_cache_dir),
        "n_splits": n_splits,
        "score_metric": score_metric,
        "fitness_random_state": fitness_random_state,
        "seed": seed,
        "population_size": population_size,
        "max_depth": max_depth,
        "use_active_set": use_active_set,
        "promotion_interval": promotion_interval,
        "first_promotion_top_k": first_promotion_top_k,
        "promotion_add_k": promotion_add_k,
        "promotion_refresh_top_n": promotion_refresh_top_n,
        "archive_quality_threshold": archive_quality_threshold,
        "archive_correlation_threshold": archive_correlation_threshold,
        "active_correlation_threshold": active_correlation_threshold,
        "promotion_min_gain": promotion_min_gain,
        "promotion_mean_gain": promotion_mean_gain,
        "feature_labels": feature_labels,
    }


def _log_generation_metrics(
    store: MlflowRunStore,
    run_id: str,
    lifecycle: SearchLifecycleRecorder | None,
) -> None:
    if lifecycle is None:
        return
    for row in lifecycle.generation_rows:
        generation = row.get("Generation")
        if isinstance(generation, bool) or not isinstance(generation, int):
            continue
        metrics = {
            metric_name: row[source_name]
            for source_name, metric_name in _GENERATION_METRICS.items()
            if source_name in row
        }
        store.log_generation_metrics(run_id, generation, metrics)  # type: ignore[arg-type]


def _store_search_failure(
    store: MlflowRunStore,
    run_id: str,
    writer: RunBundleWriter,
    error: BaseException,
    *,
    project_state: str,
) -> None:
    """Persist partial diagnostics without masking the execution exception."""

    lifecycle = writer.lifecycle
    if lifecycle is not None:
        lifecycle.close()
    try:
        partial = writer.finalize(
            "interrupted" if project_state == "interrupted" else "search_failed",
            lifecycle=lifecycle,
            error=error,
        )
        store.log_artifact_bundle(run_id, partial.path)
    except BaseException as storage_error:  # noqa: BLE001 - preserve interrupts too
        writer.cleanup()
        if hasattr(error, "add_note"):
            error.add_note(
                f"Could not persist partial MLflow artifacts: {storage_error}"
            )
    finally:
        store.terminate_run(run_id, project_state)


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
    score_metric: str = "brier_improvement",
    fitness_random_state: int = DEFAULT_RANDOM_STATE,
    seed: int = 42,
    population_size: int = 50,
    max_depth: int | None = None,
    use_active_set: bool = False,
    promotion_interval: int = 5,
    first_promotion_top_k: int = 2,
    promotion_add_k: int = 1,
    promotion_refresh_top_n: int = 50,
    archive_quality_threshold: float = 0.001,
    archive_correlation_threshold: float = 0.85,
    active_correlation_threshold: float = 0.90,
    promotion_min_gain: float = 0.0,
    promotion_mean_gain: float = 0.0005,
    feature_labels: str = "expression",
    tracking_uri: str | None = None,
    artifact_root: str | PathLike[str] | None = None,
    tracking_store: MlflowRunStore | None = None,
) -> SearchRunResult:
    """Run, analyze, and persist one canonical MLflow feature-search run."""

    selected_strategy = _coerce_strategy(strategy)
    if feature_labels not in {"id", "expression"}:
        raise ValueError("feature_labels must be 'id' or 'expression'")
    if not isinstance(use_active_set, bool):
        raise ValueError("use_active_set must be a boolean")
    if use_active_set and selected_strategy is not SearchStrategy.GENETIC:
        raise ValueError("use_active_set is supported only by the genetic strategy")
    if use_active_set and score_metric != "brier_improvement":
        raise ValueError("use_active_set requires score_metric='brier_improvement'")
    validated_time_budget, validated_candidate_count = _validate_budget_contract(
        selected_strategy,
        time_budget_seconds=time_budget_seconds,
        candidate_count=candidate_count,
    )
    if tracking_store is not None and (
        tracking_uri is not None or artifact_root is not None
    ):
        raise ValueError(
            "tracking_store cannot be combined with tracking_uri or artifact_root"
        )
    # This health probe deliberately precedes input fingerprinting and all
    # dataset, materializer, and search construction.
    store = tracking_store or MlflowRunStore(
        tracking_uri,
        artifact_root=artifact_root,
    )
    parameters = _run_parameters(
        time_budget_seconds=validated_time_budget,
        candidate_count=validated_candidate_count,
        dataset_path=dataset_path,
        mapping=mapping,
        mmap_dir=mmap_dir,
        feature_cache_dir=feature_cache_dir,
        n_splits=n_splits,
        score_metric=score_metric,
        fitness_random_state=fitness_random_state,
        seed=seed,
        population_size=population_size,
        max_depth=max_depth,
        use_active_set=use_active_set,
        promotion_interval=promotion_interval,
        first_promotion_top_k=first_promotion_top_k,
        promotion_add_k=promotion_add_k,
        promotion_refresh_top_n=promotion_refresh_top_n,
        archive_quality_threshold=archive_quality_threshold,
        archive_correlation_threshold=archive_correlation_threshold,
        active_correlation_threshold=active_correlation_threshold,
        promotion_min_gain=promotion_min_gain,
        promotion_mean_gain=promotion_mean_gain,
        feature_labels=feature_labels,
    )
    run = store.create_run(
        selected_strategy.value,
        seed,
        parameters=parameters,
        strategy_group=_strategy_group(
            selected_strategy, use_active_set=use_active_set
        ),
    )
    run_id = run.info.run_id

    with tempfile.TemporaryDirectory(prefix=f"automatedfe-{run_id}-") as temporary:
        try:
            from ..analysis.run_bundle import RunBundleWriter

            writer = RunBundleWriter(
                Path(temporary) / run_id,
                run_id=run_id,
                strategy=selected_strategy.value,
                dataset_path=dataset_path,
                mapping=mapping,
                mmap_dir=mmap_dir,
                configuration=parameters,
            )
            store.log_fingerprints(run_id, writer.inputs)
        except BaseException as error:
            store.terminate_run(
                run_id,
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "search_failed",
            )
            raise

        try:
            result = _run_feature_search_impl(
                selected_strategy,
                time_budget_seconds=validated_time_budget,
                candidate_count=validated_candidate_count,
                dataset_path=dataset_path,
                mapping=mapping,
                mmap_dir=mmap_dir,
                feature_cache_dir=feature_cache_dir,
                n_splits=n_splits,
                score_metric=score_metric,
                fitness_random_state=fitness_random_state,
                seed=seed,
                population_size=population_size,
                max_depth=max_depth,
                use_active_set=use_active_set,
                promotion_interval=promotion_interval,
                first_promotion_top_k=first_promotion_top_k,
                promotion_add_k=promotion_add_k,
                promotion_refresh_top_n=promotion_refresh_top_n,
                archive_quality_threshold=archive_quality_threshold,
                archive_correlation_threshold=archive_correlation_threshold,
                active_correlation_threshold=active_correlation_threshold,
                promotion_min_gain=promotion_min_gain,
                promotion_mean_gain=promotion_mean_gain,
                _bundle_writer=writer,
            )
            _log_generation_metrics(store, run_id, result.lifecycle)
            if result.lifecycle is not None:
                result.lifecycle.close()
            completed = writer.finalize("search_complete", lifecycle=result.lifecycle)
        except BaseException as error:
            _store_search_failure(
                store,
                run_id,
                writer,
                error,
                project_state=(
                    "interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else "search_failed"
                ),
            )
            raise

        try:
            render_run_report(completed.path, feature_labels=feature_labels)
        except KeyboardInterrupt as error:
            try:
                store.log_artifact_bundle(run_id, completed.path)
            except BaseException as storage_error:  # noqa: BLE001
                error.add_note(
                    f"Could not persist interrupted MLflow artifacts: {storage_error}"
                )
            finally:
                store.terminate_run(run_id, "interrupted")
            raise
        except BaseException as error:
            try:
                store.log_artifact_bundle(run_id, completed.path)
            except BaseException as storage_error:  # noqa: BLE001
                error.add_note(
                    f"Could not persist completed search artifacts: {storage_error}"
                )
            finally:
                store.terminate_run(run_id, "analysis_failed")
            raise SearchAnalysisError(run_id, error) from error

        try:
            store.log_artifact_bundle(run_id, completed.path)
        except BaseException:
            store.terminate_run(run_id, "analysis_failed")
            raise
        store.terminate_run(run_id, "complete")
        return replace(result, run_id=run_id)


__all__ = [
    "SearchAnalysisError",
    "SearchLifecycleRecorder",
    "SearchRunResult",
    "SearchStrategy",
    "run_feature_search",
]

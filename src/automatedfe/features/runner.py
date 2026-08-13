"""Unified orchestration for feature-search strategies.

The lower-level search builders intentionally remain available for callers
that need to customize Genetic Engine's lifecycle. This module provides the
common workflow used by experiments: configure one strategy, search, and
evaluate the resulting feature set on the held-out split.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from os import PathLike
from pathlib import Path
from time import monotonic_ns
from typing import Any, TextIO

from geneticengine.evaluation.budget import TimeBudget

from ..data.transaction_materialization import DEFAULT_MMAP_DIR
from .feature_materialization import FeatureMaterializer
from ..evaluation.final_evaluation import (
    AdditiveEvaluationResult,
    FinalEvaluationResult,
    FinalEvaluator,
)
from ..evaluation.fitness import DEFAULT_N_SPLITS, DEFAULT_RANDOM_STATE
from .grammar import collect_features, expr
from .search.search import canonical_expression_key
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

DIAGNOSTIC_COLUMNS = (
    "Strategy",
    "CandidateIndex",
    "Generation",
    "Expression",
    "Dependencies",
    "Split1",
    "Split2",
    "Split3",
    "MaterializationTime",
    "ArchiveMember",
    "Status",
    "Error",
)


class RunnerDiagnosticsRecorder:
    """Incrementally record the common runner-level candidate diagnostics."""

    def __init__(self, path: str | PathLike[str], strategy: SearchStrategy) -> None:
        self.path = Path(path).resolve()
        self.strategy = strategy
        self.rows: list[dict[str, object]] = []
        self._seen: set[str] = set()
        self._row_keys: list[str] = []
        self._file: TextIO | None = None
        self._writer: csv.DictWriter | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=DIAGNOSTIC_COLUMNS)
        self._writer.writeheader()
        self._file.flush()

    def register(
        self,
        tracker: Any,
        individual: Any,
        problem: Any,
        is_best: bool = False,
    ) -> None:
        """Record one evaluated individual; exact structural repeats are omitted."""

        del is_best
        expression = _as_expression(individual)
        key = canonical_expression_key(expression)
        if key in self._seen:
            return

        fitness = individual.get_fitness(problem)
        components = tuple(fitness.fitness_components)
        try:
            valid = (
                fitness.valid
                and len(components) == 4
                and all(math.isfinite(float(value)) for value in components)
            )
        except (TypeError, ValueError, OverflowError):
            valid = False

        error = ""
        if not valid:
            reasons = getattr(tracker.evaluator, "invalid_reasons", {})
            error = reasons.get(key, "invalid objective vector")
        generation: object = ""
        if self.strategy is SearchStrategy.GENETIC:
            generation = individual.metadata.get("generation", "")
        self._record(
            expression,
            generation=generation,
            objectives=components if valid else None,
            status="evaluated" if valid else "invalid",
            error=error,
        )

    def record_generated(self, expression: expr) -> None:
        """Record one evaluation-free enumerative expression."""

        self._record(
            expression,
            generation="",
            objectives=None,
            status="generated",
            error="",
        )

    def _record(
        self,
        expression: expr,
        *,
        generation: object,
        objectives: Sequence[object] | None,
        status: str,
        error: str,
    ) -> None:
        key = canonical_expression_key(expression)
        if key in self._seen:
            return
        self._seen.add(key)
        objective_cells: tuple[object, object, object, object]
        if objectives is None:
            objective_cells = ("", "", "", "")
        else:
            if len(objectives) != 4:
                raise ValueError("diagnostic objectives must contain four values")
            objective_cells = (
                objectives[0],
                objectives[1],
                objectives[2],
                objectives[3],
            )
        row: dict[str, object] = {
            "Strategy": self.strategy.value,
            "CandidateIndex": len(self.rows),
            "Generation": generation,
            "Expression": str(expression),
            "Dependencies": ";".join(
                sorted(feature.name for feature in collect_features(expression))
            ),
            "Split1": objective_cells[0],
            "Split2": objective_cells[1],
            "Split3": objective_cells[2],
            "MaterializationTime": objective_cells[3],
            "ArchiveMember": "",
            "Status": status,
            "Error": error,
        }
        self.rows.append(row)
        self._row_keys.append(key)
        assert self._writer is not None and self._file is not None
        self._writer.writerow(row)
        self._file.flush()

    def finalize(self, archive_expressions: Sequence[expr]) -> Path:
        """Atomically rewrite the CSV with final archive membership."""

        archive_keys = {
            canonical_expression_key(expression) for expression in archive_expressions
        }
        for row, key in zip(self.rows, self._row_keys):
            row["ArchiveMember"] = key in archive_keys
        self.close()
        return _atomic_write_diagnostics(self.path, self.rows)

    def close(self) -> None:
        """Close the incremental output, leaving it readable after failures."""

        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


def _atomic_write_diagnostics(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=DIAGNOSTIC_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary_name)
        raise
    return path


def _atomic_write_json(path: Path, document: Mapping[str, object]) -> Path:
    """Write a JSON document through a durable sibling temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            json.dump(document, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary_name)
        raise
    return path


def write_summary_json(
    path: str | PathLike[str],
    summary: Mapping[str, object],
    *,
    force: bool = False,
) -> Path:
    """Atomically persist a runner summary, protecting existing files by default."""

    if not isinstance(force, bool):
        raise ValueError("force must be a boolean")
    resolved_path = Path(path).resolve()
    if resolved_path.exists() and resolved_path.is_dir():
        raise ValueError(
            f"Output path must identify a file, not a directory: {resolved_path}"
        )
    if resolved_path.exists() and not force:
        raise FileExistsError(
            "Refusing to overwrite existing output without force=True: "
            f"{resolved_path}"
        )
    return _atomic_write_json(resolved_path, summary)


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
    def active_set_metrics(self) -> dict[str, float] | None:
        """Compatibility alias for :attr:`active_set_final_metrics`."""

        return self.active_set_final_metrics

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
    if not use_active_set and (history_path is not None or active_archive_path is not None):
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
        Path(active_archive_path).resolve()
        if active_archive_path is not None
        else None
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
            raise ValueError(f"Output path must identify a file, not a directory: {path}")
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
) -> SearchRunResult:
    """Run one strategy and always evaluate its selected features on test data.

    ``genetic``, ``enumerative``, and ``random`` use a positive wall-clock
    search budget. ``enumerative_without_archive`` uses a positive number of
    generated expressions instead and performs no search-time materialization
    or fitness evaluation. Search setup is completed before the search timer
    starts; held-out evaluation is timed separately. ``csv_path`` records the
    common incremental diagnostics, while ``archive_path`` saves an evaluated
    strategy's final Pareto archive once. In genetic active-set mode, the
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

    diagnostics: RunnerDiagnosticsRecorder | None = None
    search_started_ns = monotonic_ns()
    try:
        if resolved_csv_path is not None:
            diagnostics = RunnerDiagnosticsRecorder(
                resolved_csv_path,
                selected_strategy,
            )
            if selected_strategy in _EVALUATED_STRATEGIES:
                search.tracker.recorders.append(diagnostics)
            else:
                search.candidate_observers.append(diagnostics.record_generated)
        if selected_strategy in _EVALUATED_STRATEGIES:
            _reset_search_clock(search, search_started_ns)
        search_output = search.search()
    except BaseException:
        if diagnostics is not None:
            diagnostics.close()
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
    grammar_exhausted = bool(getattr(search, "grammar_exhausted", False))

    if diagnostics is not None:
        diagnostics.finalize(
            expressions if selected_strategy in _EVALUATED_STRATEGIES else ()
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
                "history_path and active_archive_path require an active-set "
                "manager"
            )
        if resolved_history_path is not None:
            manager.save_history(resolved_history_path, mapping=mapping)
        if resolved_active_archive_path is not None:
            manager.save_active_snapshot(resolved_active_archive_path, mapping=mapping)

    final_started_ns = monotonic_ns()
    final_evaluator = _build_final_evaluator(materializer, dataset_path, mapping)
    final_evaluation = final_evaluator.evaluate(expressions)
    final_evaluation_duration_seconds = (monotonic_ns() - final_started_ns) * 1e-9

    active_set_final_evaluation: FinalEvaluationResult | None = None
    active_set_final_evaluation_duration_seconds: float | None = None
    additive_evaluation: AdditiveEvaluationResult | None = None
    additive_evaluation_duration_seconds: float | None = None
    if active_mode and active_set_expressions:
        active_started_ns = monotonic_ns()
        active_set_final_evaluation = final_evaluator.evaluate(active_set_expressions)
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
        active_set_expressions=active_set_expressions,
        active_set_final_evaluation=active_set_final_evaluation,
        active_set_final_evaluation_duration_seconds=(
            active_set_final_evaluation_duration_seconds
        ),
        additive_evaluation=additive_evaluation,
        additive_evaluation_duration_seconds=additive_evaluation_duration_seconds,
        history_count=history_count,
        active_set_count=len(active_set_expressions),
    )


__all__ = [
    "DIAGNOSTIC_COLUMNS",
    "RunnerDiagnosticsRecorder",
    "SearchRunResult",
    "SearchStrategy",
    "run_feature_search",
    "write_summary_json",
]

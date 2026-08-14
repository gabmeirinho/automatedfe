"""Shared lifecycle and configuration for evaluated feature searches."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from time import monotonic_ns
from typing import Any, Protocol

from geneticengine.algorithms.gp.gp import GeneticProgrammingTwoPhase
from geneticengine.algorithms.gp.population import Population
from geneticengine.evaluation import Evaluator
from geneticengine.evaluation.budget import SearchBudget
from geneticengine.evaluation.recorder import CSVSearchRecorder
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import (
    Fitness,
    InvalidFitnessException,
    MultiObjectiveProblem,
    Problem,
)
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.solutions.individual import Individual, PhenotypicIndividual

from .archive import (
    ActiveSetManager,
    ArchiveStep,
    absolute_pearson_correlation,
    correlation_rejection,
    encode_expression,
    is_correlated_pairwise,
    validate_archive_quality_threshold,
    validate_correlation_threshold,
)
from ..features.feature_materialization import FeatureMaterializer
from ..features.feature_types import TxFeature
from ..evaluation.fitness import (
    DEFAULT_N_SPLITS,
    ActiveResidualEvaluator,
    MaterializationError,
    NumericalFitnessError,
    RandomForestFitness,
    ResidualEvaluator,
)
from ..features.grammar import build_grammar, collect_features, expr
from .lifecycle import (
    EVALUATED,
    INVALID,
    MATERIALIZATION_FAILED,
    SearchLifecycleRecorder,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_DEPTH = 4
ARCHIVE_MINIMIZE = [False, False, False, True]


class CandidateGenerator(Protocol):
    """Generate candidates for the shared evaluated-search lifecycle."""

    exhausted: bool

    def generate(
        self,
        previous: Sequence[PhenotypicIndividual],
        generation: int,
    ) -> Iterable[PhenotypicIndividual]: ...


@dataclass(frozen=True, slots=True)
class _SearchComponents:
    """Configured objects shared by all evaluated strategies."""

    grammar: Grammar
    representation: TreeBasedRepresentation
    materializer: Any
    fitness_evaluator: Any
    problem: MultiObjectiveProblem
    archive_step: ArchiveStep
    active_set_manager: ActiveSetManager | None
    random: NativeRandomSource
    max_depth: int


def _build_search_components(
    *,
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    mmap_dir: str | PathLike[str],
    feature_cache_dir: str | PathLike[str] | None = None,
    dataset_path: str | PathLike[str] | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    score_metric: str = "brier_improvement",
    fitness_random_state: int = 42,
    seed: int = 42,
    max_depth: int | None = None,
    archive_path: str | PathLike[str] | None = None,
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
) -> _SearchComponents:
    """Build the grammar, evaluator, problem, and archive foundation."""

    if dataset_path is None:
        raise ValueError("dataset_path is required for archive search")
    if n_splits != 3:
        raise ValueError("Archive mode requires exactly three folds")
    if use_active_set and score_metric != "brier_improvement":
        raise ValueError("use_active_set requires score_metric='brier_improvement'")
    grammar = build_grammar(mapping)
    if max_depth is None:
        max_depth = DEFAULT_MAX_DEPTH
    if max_depth <= 0:
        raise ValueError("max_depth must be positive")

    materializer = FeatureMaterializer(mmap_dir, features_dir=feature_cache_dir)
    random = NativeRandomSource(seed)
    representation = TreeBasedRepresentation(
        grammar,
        MaxDepthDecider(random, grammar, max_depth=max_depth),
    )
    archive_holder: dict[str, Any] = {}
    if score_metric in {"brier", "brier_improvement"} and use_active_set:
        fitness_evaluator = ActiveResidualEvaluator(
            materializer,
            dataset_path,
            n_splits=n_splits,
            score_metric=score_metric,
            active_provider=lambda: archive_holder["active_set"].active_individuals,
            baseline_version_provider=lambda: archive_holder["active_set"].baseline_version,
        )
    elif score_metric in {"brier", "brier_improvement"}:
        fitness_evaluator = ResidualEvaluator(
            materializer,
            dataset_path,
            n_splits=n_splits,
            score_metric=score_metric,
        )
    else:
        fitness_evaluator = RandomForestFitness(
            materializer,
            dataset_path,
            n_splits=n_splits,
            score_metric=score_metric,
            random_state=fitness_random_state,
        )

    objective_vector = getattr(fitness_evaluator, "objective_vector", None)
    if not callable(objective_vector):
        raise TypeError("Archive search requires an objective_vector evaluator")
    problem = MultiObjectiveProblem(
        fitness_function=objective_vector,
        minimize=list(ARCHIVE_MINIMIZE),
    )

    if archive_path is not None:
        resolved_archive_path = Path(archive_path).resolve()
        if resolved_archive_path.exists() and resolved_archive_path.is_dir():
            raise ValueError(
                "archive_path must identify a file, not a directory: "
                f"{resolved_archive_path}"
            )
    active_set_manager: ActiveSetManager | None = None
    if use_active_set:
        signal_provider = getattr(fitness_evaluator, "_values_for", None)
        active_set_manager = ActiveSetManager(
            signal_provider=signal_provider if callable(signal_provider) else None,
            use_active_set=True,
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
        archive_holder["active_set"] = active_set_manager

    # Every evaluated strategy uses the same canonical archive policy.  The
    # active-set manager is auxiliary state and never filters this archive.
    archive_step = ArchiveStep(archive_path=archive_path, mapping=mapping)
    return _SearchComponents(
        grammar=grammar,
        representation=representation,
        materializer=materializer,
        fitness_evaluator=fitness_evaluator,
        problem=problem,
        archive_step=archive_step,
        active_set_manager=active_set_manager,
        random=random,
        max_depth=max_depth,
    )


def canonical_expression_key(expression: object) -> str:
    """Return a stable structural key for an expression."""

    if isinstance(expression, expr):
        encoded: object = encode_expression(expression)
    elif isinstance(expression, TxFeature):
        encoded = {
            "type": "TxFeature",
            "name": expression.name,
        }
    else:
        encoded = {
            "type": f"{type(expression).__module__}.{type(expression).__qualname__}",
            "value": str(expression),
        }
    return json.dumps(encoded, sort_keys=True, separators=(",", ":"))


class CandidateEvaluator(SequentialEvaluator):
    """Turn candidate-local numerical failures into invalid fitness.

    Candidate-local materialization failures raise
    :class:`MaterializationError` and are recorded as ``materialization_failed``
    events: they never count as completed evaluations, so failed
    materializations cannot masquerade as completed candidates. Evaluations
    that complete with an invalid fitness are recorded as ``invalid`` events
    and retain the candidate's measured materialization duration when one is
    available.
    """

    def __init__(
        self,
        baseline_version_provider: Callable[[], int] | None = None,
        *,
        lifecycle: SearchLifecycleRecorder | None = None,
        materialization_duration_provider: Callable[[], float | None] | None = None,
    ) -> None:
        super().__init__()
        self.invalid_reasons: dict[str, str] = {}
        self.baseline_version_provider = baseline_version_provider
        self.lifecycle = lifecycle
        self.materialization_duration_provider = materialization_duration_provider

    @property
    def baseline_version(self) -> int | None:
        if self.baseline_version_provider is None:
            return None
        return int(self.baseline_version_provider())

    def _invalidate_if_stale(self, individual: Individual, problem: Problem) -> None:
        version = self.baseline_version
        if version is None:
            return
        if individual.metadata.get("evaluated_baseline_version") == version:
            return
        individual.fitness_store.pop(problem, None)

    def _materialization_duration(self) -> float | None:
        provider = self.materialization_duration_provider
        if provider is None:
            return None
        try:
            value = provider()
        except (TypeError, ValueError):
            return None
        if value is None or isinstance(value, bool):
            return None
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(converted) or converted < 0:
            return None
        return converted

    def evaluate_async(self, problem: Problem, individuals: Iterable[Individual]):
        for individual in individuals:
            self._invalidate_if_stale(individual, problem)
            if individual.has_fitness(problem):
                yield individual
                continue

            key = canonical_expression_key(individual.get_phenotype())
            reason = None
            status = EVALUATED
            try:
                fitness = self.eval_single(problem, individual)
            except MaterializationError as error:
                fitness = problem.get_invalid_fitness()
                reason = f"{type(error).__name__}: {error}"
                status = MATERIALIZATION_FAILED
            except (
                InvalidFitnessException,
                ArithmeticError,
                NumericalFitnessError,
            ) as error:
                fitness = problem.get_invalid_fitness()
                reason = f"{type(error).__name__}: {error}"
                status = INVALID

            components = fitness.fitness_components
            try:
                valid = (
                    fitness.valid
                    and len(components) == problem.number_of_objectives()
                    and all(math.isfinite(float(value)) for value in components)
                )
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                fitness = problem.get_invalid_fitness()
                reason = reason or "invalid objective vector"
                if status == EVALUATED:
                    status = INVALID
            if reason is not None:
                self.invalid_reasons[key] = reason

            individual.set_fitness(
                problem,
                Fitness(
                    list(fitness.fitness_components),
                    valid=fitness.valid,
                ),
            )
            version = self.baseline_version
            if version is not None:
                individual.metadata["evaluated_baseline_version"] = version
            self._record_lifecycle_evaluation(
                individual,
                problem,
                status=status,
                valid=valid,
                error=reason or "",
            )
            if status != MATERIALIZATION_FAILED:
                self.register_evaluation(individual, problem)
            yield individual

    def _record_lifecycle_evaluation(
        self,
        individual: Individual,
        problem: Problem,
        *,
        status: str,
        valid: bool,
        error: str,
    ) -> None:
        lifecycle = self.lifecycle
        if lifecycle is None:
            return
        if status == MATERIALIZATION_FAILED:
            lifecycle.on_materialization_failed(individual, error)
            return
        objectives: tuple[float, ...] | None = None
        duration: float | None = None
        if valid:
            objectives = tuple(
                float(value)
                for value in individual.get_fitness(problem).fitness_components
            )
        elif status == INVALID:
            duration = self._materialization_duration()
        lifecycle.on_candidate_evaluated(
            individual,
            status=status,
            objectives=objectives,
            duration=duration,
            error=error,
        )


class ArchiveProgressTracker(ProgressTracker):
    """Track evaluations without creating Genetic Engine's ParetoFront."""

    def __init__(
        self,
        problem: Problem,
        archive_step: ArchiveStep,
        *,
        evaluator: Evaluator | None = None,
        baseline_version_provider: Callable[[], int] | None = None,
        materialization_duration_provider: Callable[[], float | None] | None = None,
        recorders: list[object] | None = None,
    ) -> None:
        self.start_time = monotonic_ns()
        self.problem = problem
        self.evaluator = (
            evaluator
            if evaluator is not None
            else CandidateEvaluator(
                baseline_version_provider=baseline_version_provider,
                materialization_duration_provider=materialization_duration_provider,
            )
        )
        self.recorders = [] if recorders is None else recorders
        self.archive_step = archive_step
        self.memory = None

    def evaluate(self, individuals: Iterable[Individual]) -> None:
        for individual in self.evaluator.evaluate_async(self.problem, individuals):
            is_best = individual in self.archive_step.archive
            for recorder in self.recorders:
                recorder.register(
                    tracker=self,
                    individual=individual,
                    problem=self.problem,
                    is_best=is_best,
                )

    def get_best_individuals(self) -> list[Individual]:
        return list(self.archive_step.archive)


def _csv_recorder(
    csv_path: str | PathLike[str] | None,
    problem: Problem,
) -> CSVSearchRecorder | None:
    if csv_path is None:
        return None

    def phenotype(individual: PhenotypicIndividual):
        return individual.get_phenotype()

    def dependencies(individual: PhenotypicIndividual) -> str:
        return ";".join(
            sorted(feature.name for feature in collect_features(phenotype(individual)))
        )

    return CSVSearchRecorder(
        csv_path=str(csv_path),
        problem=problem,
        fields={
            "Generation": lambda _t, ind, _p: ind.metadata["generation"],
            "Expression": lambda _t, ind, _p: str(phenotype(ind)),
            "Dependencies": lambda _t, ind, _p: dependencies(ind),
            "Fitness": lambda _t, ind, p: ind.get_fitness(p).fitness_components[0],
            "Split1": lambda _t, ind, p: ind.get_fitness(p).fitness_components[0],
            "Split2": lambda _t, ind, p: ind.get_fitness(p).fitness_components[1],
            "Split3": lambda _t, ind, p: ind.get_fitness(p).fitness_components[2],
            "MaterializationTime": lambda _t, ind, p: ind.get_fitness(
                p
            ).fitness_components[3],
        },
        only_record_best_individuals=False,
    )


class MaterializingArchiveSearch(GeneticProgrammingTwoPhase):
    """Common materialize/evaluate/archive loop for evaluated strategies."""

    def __init__(
        self,
        *args: object,
        materializer: Any,
        fitness_evaluator: Any,
        archive_step: ArchiveStep,
        active_set_manager: ActiveSetManager | None = None,
        candidate_generator: CandidateGenerator | None = None,
        deduplicate: bool = False,
        lifecycle: SearchLifecycleRecorder | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.materializer = materializer
        self.fitness_evaluator = fitness_evaluator
        self.archive_step = archive_step
        self.archive = archive_step
        self.active_set_manager = active_set_manager
        self.candidate_generator = candidate_generator
        self.deduplicate = deduplicate
        self.lifecycle = (
            lifecycle if lifecycle is not None else SearchLifecycleRecorder()
        )
        self._seen: set[str] = set()
        self.generated_count = 0
        self.duplicate_count = 0
        self.accepted_count = 0
        self.invalid_count = 0
        self.accepted_individuals: list[PhenotypicIndividual] = []
        self.last_individuals: list[PhenotypicIndividual] = []
        self.history: list[PhenotypicIndividual] = []
        self.active_individuals: list[PhenotypicIndividual] = []
        self._promotion_boundaries: set[int] = set()

    @property
    def grammar_exhausted(self) -> bool:
        return bool(
            self.candidate_generator is not None
            and self.candidate_generator.exhausted
        )

    @property
    def seen_keys(self) -> frozenset[str]:
        return frozenset(self._seen)

    @property
    def invalid_reasons(self) -> dict[str, str]:
        return dict(getattr(self.tracker.evaluator, "invalid_reasons", {}))

    def _generate_initial_individuals(self) -> list[PhenotypicIndividual]:
        if self.candidate_generator is None:
            return super()._generate_initial_individuals()
        return self._generate_candidates([], 0)

    def _generate_next_individuals(
        self,
        current_individuals: list[PhenotypicIndividual],
        generation: int,
    ) -> list[PhenotypicIndividual]:
        if self.candidate_generator is None:
            return super()._generate_next_individuals(
                current_individuals,
                generation,
            )
        return self._generate_candidates(current_individuals, generation)

    def _invalidate_population_fitness(
        self,
        individuals: Iterable[PhenotypicIndividual],
    ) -> None:
        """Drop fitness copied into a population before an active refresh."""

        for individual in individuals:
            individual.fitness_store.pop(self.problem, None)

    def _promote_at_boundary(
        self,
        generation: int,
        stale_individuals: Iterable[PhenotypicIndividual] = (),
    ) -> bool:
        """Run one idempotent active promotion boundary."""

        if generation in self._promotion_boundaries:
            return False
        self._promotion_boundaries.add(generation)
        if self.active_set_manager is None:
            return False
        changed = self.active_set_manager.maybe_promote(
            self.problem,
            generation,
            evaluator=self.tracker.evaluator,
        )
        if changed:
            if hasattr(self.fitness_evaluator, "invalidate_baseline_cache"):
                self.fitness_evaluator.invalidate_baseline_cache()
            # Population and archive entries can share individual objects.
            # Clear stale population fitness first, then leave every archived
            # object carrying its freshly calculated baseline-version score.
            self._invalidate_population_fitness(stale_individuals)
            self.archive_step.reevaluate_archive(
                self.problem,
                self.tracker.evaluator,
            )
        return bool(changed)

    def _generate_candidates(
        self,
        current_individuals: list[PhenotypicIndividual],
        generation: int,
    ) -> list[PhenotypicIndividual]:
        assert self.candidate_generator is not None
        individuals = list(
            self.candidate_generator.generate(current_individuals, generation)
        )
        for individual in individuals:
            individual.metadata["generation"] = generation
        return individuals

    def _accept_candidates(
        self,
        individuals: list[PhenotypicIndividual],
    ) -> list[PhenotypicIndividual]:
        accepted: list[PhenotypicIndividual] = []
        for individual in individuals:
            self.generated_count += 1
            self.lifecycle.on_candidate_generated(individual)
            key = canonical_expression_key(individual.get_phenotype())
            if self.deduplicate and key in self._seen:
                self.duplicate_count += 1
                self.lifecycle.on_candidate_duplicate(individual)
                continue
            self._seen.add(key)
            self.accepted_count += 1
            accepted.append(individual)
        return accepted

    def perform_search(self) -> list[Individual] | None:
        lifecycle = self.lifecycle
        lifecycle.on_search_started()
        evaluator = getattr(self.tracker, "evaluator", None)
        if hasattr(evaluator, "lifecycle"):
            evaluator.lifecycle = lifecycle
        generation = 0
        current_individuals: list[PhenotypicIndividual] = []

        while generation == 0 or not self.is_done():
            lifecycle.on_generation_started(generation)
            if generation > 0:
                self._promote_at_boundary(
                    generation,
                    stale_individuals=current_individuals,
                )
            generated = (
                self._generate_initial_individuals()
                if generation == 0
                else self._generate_next_individuals(
                    current_individuals,
                    generation,
                )
            )
            accepted = self._accept_candidates(generated)
            if not accepted:
                if self.grammar_exhausted:
                    break
                generation += 1
                continue
            accepted = self.precompute_population(accepted, generation)
            if not accepted:
                if self.grammar_exhausted:
                    break
                generation += 1
                continue

            archived = list(
                self.archive_step.apply(
                    self.problem,
                    self.tracker.evaluator,
                    self.representation,
                    self.random,
                    iter(accepted),
                    len(accepted),
                    generation,
                )
            )
            if self.active_set_manager is not None:
                self.active_set_manager.process_evaluated_population(
                    self.problem,
                    archived,
                    generation,
                )
            current_population = Population(
                iter(archived),
                self.tracker,
                generation=generation,
            )
            current_individuals = current_population.get_individuals()
            self.accepted_individuals.extend(current_individuals)
            self.invalid_count += sum(
                not individual.get_fitness(self.problem).valid
                for individual in current_individuals
            )
            # Persist the permanent archive, not the temporary front used by
            # the current generation's admission gate.
            lifecycle.on_generation_completed(
                generation,
                self.archive_step.archive_snapshot(),
            )
            generation += 1

        if self.active_set_manager is not None:
            self.history = list(self.active_set_manager.history)
            self.active_individuals = list(
                self.active_set_manager.active_individuals
            )
        lifecycle.on_search_completed(
            canonical_expression_key(individual.get_phenotype())
            for individual in self.archive_step.archive
        )
        return list(self.archive_step.archive)

    def precompute_population(
        self,
        individuals: list[PhenotypicIndividual],
        generation: int,
    ) -> list[PhenotypicIndividual]:
        """Prepare generated individuals before their evaluation.

        Candidates that fail to materialize are recorded as
        ``materialization_failed`` and excluded from the generation: they are
        never evaluated and never reach the archive. Returns the
        materializable subset in the original order.
        """

        self._promote_at_boundary(
            generation,
            stale_individuals=individuals,
        )
        keep: list[PhenotypicIndividual] = []
        for individual in individuals:
            try:
                self.fitness_evaluator.prepare_population(
                    [individual.get_phenotype()]
                )
            except MaterializationError as error:
                self.lifecycle.on_materialization_failed(
                    individual,
                    error=str(error),
                )
            else:
                keep.append(individual)
        self.last_individuals = list(keep)
        logger.info(
            "Materializing generation %d: %d features",
            generation,
            len(keep),
        )
        return keep


def _build_evaluated_search(
    budget: SearchBudget,
    *,
    candidate_generator_factory: Callable[[_SearchComponents], CandidateGenerator],
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    mmap_dir: str | PathLike[str],
    feature_cache_dir: str | PathLike[str] | None = None,
    dataset_path: str | PathLike[str] | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    score_metric: str = "brier_improvement",
    fitness_random_state: int = 42,
    seed: int = 42,
    max_depth: int | None = None,
    csv_path: str | PathLike[str] | None = None,
    archive_path: str | PathLike[str] | None = None,
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
) -> MaterializingArchiveSearch:
    """Build a candidate-generating strategy on the shared lifecycle."""

    components = _build_search_components(
        mapping=mapping,
        mmap_dir=mmap_dir,
        feature_cache_dir=feature_cache_dir,
        dataset_path=dataset_path,
        n_splits=n_splits,
        score_metric=score_metric,
        fitness_random_state=fitness_random_state,
        seed=seed,
        max_depth=max_depth,
        archive_path=archive_path,
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
    candidate_generator = candidate_generator_factory(components)
    recorder = _csv_recorder(csv_path, components.problem)
    tracker = ArchiveProgressTracker(
        components.problem,
        components.archive_step,
        baseline_version_provider=(
            (lambda: components.active_set_manager.baseline_version)
            if components.active_set_manager is not None
            else None
        ),
        materialization_duration_provider=(
            lambda: getattr(
                components.fitness_evaluator,
                "last_materialization_duration",
                None,
            )
        ),
        recorders=[] if recorder is None else [recorder],
    )
    return MaterializingArchiveSearch(
        problem=components.problem,
        budget=budget,
        representation=components.representation,
        population_size=1,
        random=components.random,
        tracker=tracker,
        materializer=components.materializer,
        fitness_evaluator=components.fitness_evaluator,
        archive_step=components.archive_step,
        active_set_manager=components.active_set_manager,
        candidate_generator=candidate_generator,
        deduplicate=True,
    )


__all__ = [
    "ARCHIVE_MINIMIZE",
    "DEFAULT_MAX_DEPTH",
    "ArchiveProgressTracker",
    "ActiveSetManager",
    "CandidateEvaluator",
    "CandidateGenerator",
    "MaterializingArchiveSearch",
    "SearchLifecycleRecorder",
    "_SearchComponents",
    "_build_evaluated_search",
    "_build_search_components",
    "_csv_recorder",
    "absolute_pearson_correlation",
    "canonical_expression_key",
    "correlation_rejection",
    "is_correlated_pairwise",
    "validate_archive_quality_threshold",
    "validate_correlation_threshold",
]

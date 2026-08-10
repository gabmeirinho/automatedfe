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

from ..archive import ArchiveStep, encode_expression
from ..feature_materialization import FeatureMaterializer
from ..feature_types import TxFeature
from ..fitness import (
    DEFAULT_N_SPLITS,
    NumericalFitnessError,
    RandomForestFitness,
    ResidualEvaluator,
)
from ..grammar import build_grammar, collect_features, expr

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
    random: NativeRandomSource
    max_depth: int


def _build_search_components(
    *,
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    mmap_dir: str | PathLike[str],
    feature_cache_dir: str | PathLike[str] | None = None,
    dataset_path: str | PathLike[str] | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    score_metric: str = "roc_auc",
    fitness_random_state: int = 42,
    seed: int = 42,
    max_depth: int | None = None,
    archive_path: str | PathLike[str] | None = None,
) -> _SearchComponents:
    """Build the grammar, evaluator, problem, and archive foundation."""

    if dataset_path is None:
        raise ValueError("dataset_path is required for archive search")
    if n_splits != 3:
        raise ValueError("Archive mode requires exactly three folds")
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
    if score_metric in {"brier", "brier_improvement"}:
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
    archive_step = ArchiveStep(archive_path=archive_path, mapping=mapping)
    return _SearchComponents(
        grammar=grammar,
        representation=representation,
        materializer=materializer,
        fitness_evaluator=fitness_evaluator,
        problem=problem,
        archive_step=archive_step,
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
    """Turn candidate-local numerical failures into invalid fitness."""

    def __init__(self) -> None:
        super().__init__()
        self.invalid_reasons: dict[str, str] = {}

    def evaluate_async(self, problem: Problem, individuals: Iterable[Individual]):
        for individual in individuals:
            if individual.has_fitness(problem):
                yield individual
                continue

            key = canonical_expression_key(individual.get_phenotype())
            reason = None
            try:
                fitness = self.eval_single(problem, individual)
            except (
                InvalidFitnessException,
                ArithmeticError,
                NumericalFitnessError,
            ) as error:
                fitness = problem.get_invalid_fitness()
                reason = f"{type(error).__name__}: {error}"

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
            if reason is not None:
                self.invalid_reasons[key] = reason

            individual.set_fitness(
                problem,
                Fitness(
                    list(fitness.fitness_components),
                    valid=fitness.valid,
                ),
            )
            self.register_evaluation(individual, problem)
            yield individual


class ArchiveProgressTracker(ProgressTracker):
    """Track evaluations without creating Genetic Engine's ParetoFront."""

    def __init__(
        self,
        problem: Problem,
        archive_step: ArchiveStep,
        *,
        evaluator: Evaluator | None = None,
        recorders: list[object] | None = None,
    ) -> None:
        self.start_time = monotonic_ns()
        self.problem = problem
        self.evaluator = evaluator if evaluator is not None else CandidateEvaluator()
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
        candidate_generator: CandidateGenerator | None = None,
        deduplicate: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.materializer = materializer
        self.fitness_evaluator = fitness_evaluator
        self.archive_step = archive_step
        self.archive = archive_step
        self.candidate_generator = candidate_generator
        self.deduplicate = deduplicate
        self._seen: set[str] = set()
        self.generated_count = 0
        self.duplicate_count = 0
        self.accepted_count = 0
        self.invalid_count = 0
        self.accepted_individuals: list[PhenotypicIndividual] = []
        self.last_individuals: list[PhenotypicIndividual] = []

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
            key = canonical_expression_key(individual.get_phenotype())
            if self.deduplicate and key in self._seen:
                self.duplicate_count += 1
                continue
            self._seen.add(key)
            self.accepted_count += 1
            accepted.append(individual)
        return accepted

    def perform_search(self) -> list[Individual] | None:
        generation = 0
        current_individuals: list[PhenotypicIndividual] = []

        while generation == 0 or not self.is_done():
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

            self.precompute_population(accepted, generation)
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
            generation += 1

        return list(self.archive_step.archive)

    def precompute_population(
        self,
        individuals: list[PhenotypicIndividual],
        generation: int,
    ) -> None:
        """Prepare generated individuals before their evaluation."""

        self.last_individuals = list(individuals)
        phenotypes = [individual.get_phenotype() for individual in individuals]
        logger.info(
            "Materializing generation %d: %d features",
            generation,
            len(phenotypes),
        )
        self.fitness_evaluator.prepare_population(phenotypes)


def _build_evaluated_search(
    budget: SearchBudget,
    *,
    candidate_generator_factory: Callable[[_SearchComponents], CandidateGenerator],
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    mmap_dir: str | PathLike[str],
    feature_cache_dir: str | PathLike[str] | None = None,
    dataset_path: str | PathLike[str] | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    score_metric: str = "roc_auc",
    fitness_random_state: int = 42,
    seed: int = 42,
    max_depth: int | None = None,
    csv_path: str | PathLike[str] | None = None,
    archive_path: str | PathLike[str] | None = None,
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
    )
    candidate_generator = candidate_generator_factory(components)
    recorder = _csv_recorder(csv_path, components.problem)
    tracker = ArchiveProgressTracker(
        components.problem,
        components.archive_step,
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
        candidate_generator=candidate_generator,
        deduplicate=True,
    )


__all__ = [
    "ARCHIVE_MINIMIZE",
    "DEFAULT_MAX_DEPTH",
    "ArchiveProgressTracker",
    "CandidateEvaluator",
    "CandidateGenerator",
    "MaterializingArchiveSearch",
    "_SearchComponents",
    "_build_evaluated_search",
    "_build_search_components",
    "_csv_recorder",
    "canonical_expression_key",
]

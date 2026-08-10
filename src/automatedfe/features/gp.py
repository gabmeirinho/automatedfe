"""Search configuration for the feature-search genetic program."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from time import monotonic_ns
from typing import Protocol

from geneticengine.algorithms.gp.gp import (
    GeneticProgramming,
    GeneticProgrammingTwoPhase,
)
from geneticengine.algorithms.gp.population import Population
from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.selection import LexicaseSelection
from geneticengine.evaluation.budget import SearchBudget
from geneticengine.evaluation import Evaluator
from geneticengine.evaluation.recorder import CSVSearchRecorder
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.problems import (
    Fitness,
    InvalidFitnessException,
    MultiObjectiveProblem,
    Problem,
)
from geneticengine.solutions.individual import Individual, PhenotypicIndividual

from .archive import ArchiveStep
from .feature_materialization import FeatureMaterializer
from .fitness import (
    DEFAULT_N_SPLITS,
    NumericalFitnessError,
    RandomForestFitness,
    ResidualEvaluator,
)
from .grammar import build_grammar, collect_features
from .search_strategies import _build_search_components, canonical_expression_key

logger = logging.getLogger(__name__)


class CandidateGenerator(Protocol):
    """The only behavior that differs between evaluated search strategies."""

    exhausted: bool

    def generate(
        self,
        previous: Sequence[PhenotypicIndividual],
        generation: int,
    ) -> Iterable[PhenotypicIndividual]: ...


class CandidateEvaluator(SequentialEvaluator):
    """Turn explicitly candidate-local numerical failures into invalid fitness."""

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


def _multiobjective_programming_step(archive_step: ArchiveStep) -> SequenceStep:
    """Build the GP generation pipeline used by archive mode."""

    return SequenceStep(
        # ArchiveStep receives the already-evaluated current population before
        # the next generation is produced. This preserves the two-phase
        # materialization lifecycle.
        archive_step,
        ParallelStep(
            [
                ElitismStep(),
                NoveltyStep(),
                SequenceStep(
                    LexicaseSelection(epsilon=True),
                    GenericCrossoverStep(0.01),
                    GenericMutationStep(0.9),
                ),
            ],
            weights=[5, 5, 90],
        ),
    )


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


def build_search_algorithm(
    budget: SearchBudget,
    *,
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    population_size: int = 20,
    seed: int = 42,
    csv_path: str | PathLike[str] | None = None,
    archive_path: str | PathLike[str] | None = None,
    mmap_dir: str | PathLike[str],
    feature_cache_dir: str | PathLike[str] | None = None,
    dataset_path: str | PathLike[str] | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    score_metric: str = "roc_auc",
    fitness_random_state: int = 42,
    max_depth: int | None = None,
) -> GeneticProgramming:
    """Configure the GP search over the complete expression grammar.

    *mapping* supplies the encoded category values used by categorical
    terminals. When omitted, the persisted preprocessing mapping is loaded.

    *feature_cache_dir* stores the event-level feature values (one ``float64``
    per event) computed during the search. Features already present in the
    cache are loaded from disk instead of recomputed, so repeated runs over
    the same event set reuse previous work. Archive mode evaluates each
    generated feature on exactly three
    chronological cross-validation folds and uses the resulting objective
    vector plus materialization time. The default metrics use a fresh
    random-forest fit defined by :class:`RandomForestFitness`;
    ``score_metric='brier_improvement'`` (or ``'brier'``) selects the cheap
    intercept-plus-residual evaluator defined by :class:`ResidualEvaluator`.
    A dataset path is required because this search is always multiobjective.
    When *archive_path* is supplied, the current strict Pareto front is saved
    atomically as a JSON snapshot after each completed generation.
    """

    if population_size <= 0:
        raise ValueError("population_size must be positive")
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
        # Keep these module-level names as factories so the legacy builder's
        # existing test and extension points remain intact.
        materializer_factory=FeatureMaterializer,
        random_fitness_factory=RandomForestFitness,
        residual_fitness_factory=ResidualEvaluator,
    )
    grammar = components.grammar
    materializer = components.materializer
    random = components.random
    representation = components.representation
    fitness_evaluator = components.fitness_evaluator
    problem = components.problem
    archive_step = components.archive_step
    generation_step = _multiobjective_programming_step(archive_step)

    recorder = _csv_recorder(csv_path, problem)
    tracker = ArchiveProgressTracker(
        problem,
        archive_step,
        recorders=[] if recorder is None else [recorder],
    )

    return MaterializingGeneticProgramming(
        problem=problem,
        budget=budget,
        representation=representation,
        population_size=population_size,
        random=random,
        tracker=tracker,
        materializer=materializer,
        fitness_evaluator=fitness_evaluator,
        archive_step=archive_step,
        step=generation_step,
    )


class MaterializingArchiveSearch(GeneticProgrammingTwoPhase):
    """Common materialize/evaluate/archive loop for evaluated strategies.

    Genetic search uses Genetic Engine's population initializer and evolution
    step. Enumerative and random search inject a candidate generator instead;
    everything after candidate creation follows this same lifecycle.
    """

    def __init__(
        self,
        *args: object,
        materializer: FeatureMaterializer,
        fitness_evaluator: RandomForestFitness | ResidualEvaluator,
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
        """Prepare the already-generated individuals before evaluation.

        The evaluator prepares the event-level feature values, which are
        computed and cached on disk before the population is evaluated.
        """

        self.last_individuals = list(individuals)
        phenotypes = [individual.get_phenotype() for individual in individuals]
        logger.info(
            "Materializing generation %d: %d features",
            generation,
            len(phenotypes),
        )
        self.fitness_evaluator.prepare_population(phenotypes)


class MaterializingGeneticProgramming(MaterializingArchiveSearch):
    """Backward-compatible name for the genetic candidate strategy."""


__all__ = [
    "MaterializingArchiveSearch",
    "MaterializingGeneticProgramming",
    "build_grammar",
    "build_search_algorithm",
]

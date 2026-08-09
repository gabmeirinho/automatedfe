"""Search configuration for the feature-search genetic program."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path
from time import monotonic_ns

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
from geneticengine.problems import MultiObjectiveProblem, Problem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.solutions.individual import Individual, PhenotypicIndividual

from .archive import ArchiveStep
from .feature_materialization import FeatureMaterializer
from .fitness import DEFAULT_N_SPLITS, LogisticRegressionFitness, ResidualEvaluator
from .grammar import build_grammar, collect_features

logger = logging.getLogger(__name__)


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
        self.evaluator = evaluator if evaluator is not None else SequentialEvaluator()
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
    logistic-regression fit defined by :class:`LogisticRegressionFitness`;
    ``score_metric='brier_improvement'`` (or ``'brier'``) selects the cheap
    intercept-plus-residual evaluator defined by :class:`ResidualEvaluator`.
    A dataset path is required because this search is always multiobjective.
    When *archive_path* is supplied, the current strict Pareto front is saved
    atomically as a JSON snapshot after each completed generation.
    """

    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if dataset_path is None:
        raise ValueError("dataset_path is required for archive search")
    if n_splits != 3:
        raise ValueError("Archive mode requires exactly three folds")
    grammar = build_grammar(mapping)
    if max_depth is None:
        max_depth = 4
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
        fitness_evaluator = LogisticRegressionFitness(
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
        minimize=[False, False, False, True],
    )
    if archive_path is not None:
        resolved_archive_path = Path(archive_path).resolve()
        if resolved_archive_path.exists() and resolved_archive_path.is_dir():
            raise ValueError(
                "archive_path must identify a file, not a directory: "
                f"{resolved_archive_path}"
            )
    archive_step = ArchiveStep(archive_path=archive_path, mapping=mapping)
    generation_step = _multiobjective_programming_step(archive_step)

    recorder = None
    if csv_path is not None:
        def _phenotype(individual: PhenotypicIndividual):
            return individual.get_phenotype()

        def _dependencies(individual: PhenotypicIndividual) -> str:
            phenotype = _phenotype(individual)
            return ";".join(
                sorted(feature.name for feature in collect_features(phenotype))
            )

        recorder = CSVSearchRecorder(
            csv_path=str(csv_path),
            problem=problem,
            fields={
                "Generation": lambda _t, individual, _p: individual.metadata["generation"],
                "Expression": lambda _t, individual, _p: str(_phenotype(individual)),
                "Dependencies": lambda _t, individual, _p: _dependencies(individual),
                # Keep the historical single-objective "Fitness" column first so
                # existing consumers keep working, then expose the full four-objective
                # vector used by archive search.
                "Fitness": lambda _t, individual, p: individual.get_fitness(p).fitness_components[0],
                "Split1": lambda _t, individual, p: individual.get_fitness(p).fitness_components[0],
                "Split2": lambda _t, individual, p: individual.get_fitness(p).fitness_components[1],
                "Split3": lambda _t, individual, p: individual.get_fitness(p).fitness_components[2],
                "MaterializationTime": lambda _t, individual, p: individual.get_fitness(p).fitness_components[3],
            },
            only_record_best_individuals=False,
        )
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


class MaterializingGeneticProgramming(GeneticProgrammingTwoPhase):
    """Two-phase GP that materializes a complete generation before fitness."""

    def __init__(
        self,
        *args: object,
        materializer: FeatureMaterializer,
        fitness_evaluator: LogisticRegressionFitness | ResidualEvaluator,
        archive_step: ArchiveStep,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.materializer = materializer
        self.fitness_evaluator = fitness_evaluator
        self.archive_step = archive_step
        self.archive = archive_step
        self.last_individuals: list[PhenotypicIndividual] = []

    def perform_search(self) -> list[Individual] | None:
        generation = 0
        current_individuals = self._generate_initial_individuals()
        self.precompute_population(current_individuals, generation)
        current_individuals = list(
            self.archive_step.apply(
                self.problem,
                self.tracker.evaluator,
                self.representation,
                self.random,
                iter(current_individuals),
                len(current_individuals),
                generation,
            )
        )
        current_population = Population(
            iter(current_individuals),
            self.tracker,
            generation=generation,
        )

        while not self.is_done():
            generation += 1
            next_individuals = self._generate_next_individuals(
                current_population.get_individuals(),
                generation,
            )
            self.precompute_population(next_individuals, generation)
            current_population = Population(
                iter(next_individuals),
                self.tracker,
                generation,
            )

        # The step runs before generation production, so explicitly archive
        # the final evaluated population after the budget stops the loop.
        list(
            self.archive_step.apply(
                self.problem,
                self.tracker.evaluator,
                self.representation,
                self.random,
                iter(current_population.get_individuals()),
                len(current_population.get_individuals()),
                generation,
            )
        )
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
            "Materializing generation %d: %d GP features",
            generation,
            len(phenotypes),
        )
        self.fitness_evaluator.prepare_population(phenotypes)


__all__ = [
    "MaterializingGeneticProgramming",
    "build_grammar",
    "build_search_algorithm",
]

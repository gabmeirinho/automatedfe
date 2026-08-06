"""Search configuration for the feature-search genetic program."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from os import PathLike

from geneticengine.algorithms.gp.gp import (
    GeneticProgramming,
    GeneticProgrammingTwoPhase,
)
from geneticengine.evaluation.budget import SearchBudget
from geneticengine.evaluation.recorder import CSVSearchRecorder
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.solutions.individual import PhenotypicIndividual

from .feature_materialization import FeatureMaterializer
from .fitness import DEFAULT_N_SPLITS, LogisticRegressionFitness
from .grammar import build_grammar, collect_features

logger = logging.getLogger(__name__)


def build_search_algorithm(
    budget: SearchBudget,
    *,
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    population_size: int = 20,
    seed: int = 42,
    csv_path: str | PathLike[str] | None = None,
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
    the same event set reuse previous work. If *dataset_path* is supplied,
    each generated feature is evaluated with a fresh logistic-regression fit
    on the chronological cross-validation folds defined by
    :class:`LogisticRegressionFitness`. Leaving it ``None`` preserves the
    zero-fitness configuration used by materialization-only callers and older
    experiments.
    """

    if population_size <= 0:
        raise ValueError("population_size must be positive")
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
    fitness_evaluator = None
    if dataset_path is not None:
        fitness_evaluator = LogisticRegressionFitness(
            materializer,
            dataset_path,
            n_splits=n_splits,
            score_metric=score_metric,
            random_state=fitness_random_state,
        )
    problem = SingleObjectiveProblem(
        fitness_function=(
            fitness_evaluator if fitness_evaluator is not None else lambda _individual: 0.0
        ),
        minimize=False,
    )
    tracker = None
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
                "Fitness": lambda _t, individual, p: individual.get_fitness(p).fitness_components[0],
            },
            only_record_best_individuals=False,
        )
        tracker = ProgressTracker(problem, recorders=[recorder])

    return MaterializingGeneticProgramming(
        problem=problem,
        budget=budget,
        representation=representation,
        population_size=population_size,
        random=random,
        tracker=tracker,
        materializer=materializer,
        fitness_evaluator=fitness_evaluator,
    )


class MaterializingGeneticProgramming(GeneticProgrammingTwoPhase):
    """Two-phase GP that materializes a complete generation before fitness."""

    def __init__(
        self,
        *args: object,
        materializer: FeatureMaterializer,
        fitness_evaluator: LogisticRegressionFitness | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.materializer = materializer
        self.fitness_evaluator = fitness_evaluator
        self.last_individuals: list[PhenotypicIndividual] = []

    def precompute_population(
        self,
        individuals: list[PhenotypicIndividual],
        generation: int,
    ) -> None:
        """Prepare the already-generated individuals before evaluation.

        With a fitness evaluator, only the event-level feature values are
        prepared (computed and cached on disk); the per-transaction feature
        pass runs exclusively in the materialization-only configuration.
        """

        self.last_individuals = list(individuals)
        phenotypes = [individual.get_phenotype() for individual in individuals]
        logger.info(
            "Materializing generation %d: %d GP features",
            generation,
            len(phenotypes),
        )
        if self.fitness_evaluator is not None:
            self.fitness_evaluator.prepare_population(phenotypes)
        else:
            self.materializer.materialize_population(phenotypes)


__all__ = [
    "MaterializingGeneticProgramming",
    "build_grammar",
    "build_search_algorithm",
]

"""Search configuration for the feature-search genetic program."""

from __future__ import annotations

from os import PathLike

from geneticengine.algorithms.gp.gp import (
    GeneticProgramming,
    GeneticProgrammingTwoPhase,
)
from geneticengine.evaluation.budget import TimeBudget
from geneticengine.evaluation.recorder import CSVSearchRecorder
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.solutions.individual import PhenotypicIndividual

from .feature_materialization import FeatureMaterializer
from .grammar import Feature, Mean, WindowIndex, build_grammar


def build_search_algorithm(
    grammar: Grammar,
    budget: TimeBudget,
    *,
    population_size: int = 20,
    seed: int = 42,
    csv_path: str | PathLike[str] | None = None,
    mmap_dir: str | PathLike[str],
    feature_output_dir: str | PathLike[str] | None = None,
) -> GeneticProgramming:
    """Configure the materializing GP search."""

    if population_size <= 0:
        raise ValueError("population_size must be positive")
    materializer = FeatureMaterializer(mmap_dir, output_dir=feature_output_dir)

    random = NativeRandomSource(seed)
    representation = TreeBasedRepresentation(
        grammar,
        MaxDepthDecider(random, grammar, max_depth=1),
    )
    problem = SingleObjectiveProblem(
        fitness_function=lambda _individual: 0.0,
        minimize=False,
    )
    tracker = None
    if csv_path is not None:
        recorder = CSVSearchRecorder(
            csv_path=str(csv_path),
            problem=problem,
            fields={
                "Generation": lambda _t, individual, _p: individual.metadata["generation"],
                "Expression": lambda _t, individual, _p: individual.get_phenotype(),
                "Feature": lambda _t, individual, _p: individual.get_phenotype().feature,
                "Window": lambda _t, individual, _p: individual.get_phenotype().selected_window.name,
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
    )


class MaterializingGeneticProgramming(GeneticProgrammingTwoPhase):
    """Two-phase GP that materializes a complete generation before fitness."""

    def __init__(
        self,
        *args: object,
        materializer: FeatureMaterializer,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.materializer = materializer

    def precompute_population(
        self,
        individuals: list[PhenotypicIndividual],
        generation: int,
    ) -> None:
        """Materialize all already-generated individuals in this generation."""

        del generation
        phenotypes = [individual.get_phenotype() for individual in individuals]
        self.materializer.materialize_population(phenotypes)


__all__ = [
    "Feature",
    "MaterializingGeneticProgramming",
    "Mean",
    "WindowIndex",
    "build_grammar",
    "build_search_algorithm",
]

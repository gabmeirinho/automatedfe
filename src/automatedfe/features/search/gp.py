"""Genetic-programming search over the complete feature grammar."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.selection import LexicaseSelection
from geneticengine.evaluation.budget import SearchBudget

from ..fitness import DEFAULT_N_SPLITS
from ..grammar import build_grammar
from .search import (
    ArchiveProgressTracker,
    MaterializingArchiveSearch,
    _build_search_components,
    _csv_recorder,
)


def _multiobjective_programming_step(archive_step):
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


def build_search_algorithm(
    budget: SearchBudget,
    *,
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    population_size: int = 50,
    seed: int = 42,
    csv_path: str | PathLike[str] | None = None,
    archive_path: str | PathLike[str] | None = None,
    mmap_dir: str | PathLike[str],
    feature_cache_dir: str | PathLike[str] | None = None,
    dataset_path: str | PathLike[str] | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    score_metric: str = "brier_improvement",
    fitness_random_state: int = 42,
    max_depth: int | None = None,
) -> GeneticProgramming:
    """Build the genetic-programming search strategy."""

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
    )
    generation_step = _multiobjective_programming_step(components.archive_step)
    recorder = _csv_recorder(csv_path, components.problem)
    tracker = ArchiveProgressTracker(
        components.problem,
        components.archive_step,
        recorders=[] if recorder is None else [recorder],
    )

    return MaterializingGeneticProgramming(
        problem=components.problem,
        budget=budget,
        representation=components.representation,
        population_size=population_size,
        random=components.random,
        tracker=tracker,
        materializer=components.materializer,
        fitness_evaluator=components.fitness_evaluator,
        archive_step=components.archive_step,
        step=generation_step,
    )


class MaterializingGeneticProgramming(MaterializingArchiveSearch):
    """Genetic-programming candidate strategy using the shared lifecycle."""


__all__ = [
    "MaterializingArchiveSearch",
    "MaterializingGeneticProgramming",
    "build_grammar",
    "build_search_algorithm",
]

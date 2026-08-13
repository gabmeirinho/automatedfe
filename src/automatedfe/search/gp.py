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

from ..evaluation.fitness import DEFAULT_N_SPLITS
from ..features.grammar import build_grammar
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
    promotion_corr_threshold_active: float | None = None,
    promotion_min_delta_threshold: float | None = None,
    promotion_min_mean_delta_threshold: float | None = None,
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
        promotion_corr_threshold_active=promotion_corr_threshold_active,
        promotion_min_delta_threshold=promotion_min_delta_threshold,
        promotion_min_mean_delta_threshold=promotion_min_mean_delta_threshold,
    )
    generation_step = _multiobjective_programming_step(components.archive_step)
    recorder = _csv_recorder(csv_path, components.problem)
    tracker = ArchiveProgressTracker(
        components.problem,
        components.archive_step,
        baseline_version_provider=(
            (lambda: components.active_set_manager.baseline_version)
            if components.active_set_manager is not None
            else None
        ),
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
        active_set_manager=components.active_set_manager,
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

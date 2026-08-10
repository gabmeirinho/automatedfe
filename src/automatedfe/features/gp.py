"""Compatibility facade for the relocated GP search implementation.

The implementation lives in :mod:`automatedfe.features.search.gp`.  Keeping
this small facade preserves imports used by existing callers while ensuring
the new search package never depends on this legacy module.
"""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import SearchBudget

from .search import gp as _implementation
from .search import search as _shared
from .search.gp import MaterializingGeneticProgramming
from .search.search import (
    ArchiveProgressTracker,
    CandidateEvaluator,
    CandidateGenerator,
    MaterializingArchiveSearch,
    _build_search_components,
    _csv_recorder,
    canonical_expression_key,
)
from .feature_materialization import FeatureMaterializer
from .fitness import DEFAULT_N_SPLITS, RandomForestFitness, ResidualEvaluator
from .grammar import build_grammar

_multiobjective_programming_step = _implementation._multiobjective_programming_step


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
    """Build the relocated GP search, honoring legacy factory patch points."""

    original_factories = (
        _shared.FeatureMaterializer,
        _shared.RandomForestFitness,
        _shared.ResidualEvaluator,
    )
    _shared.FeatureMaterializer = FeatureMaterializer
    _shared.RandomForestFitness = RandomForestFitness
    _shared.ResidualEvaluator = ResidualEvaluator
    try:
        return _implementation.build_search_algorithm(
            budget,
            mapping=mapping,
            population_size=population_size,
            seed=seed,
            csv_path=csv_path,
            archive_path=archive_path,
            mmap_dir=mmap_dir,
            feature_cache_dir=feature_cache_dir,
            dataset_path=dataset_path,
            n_splits=n_splits,
            score_metric=score_metric,
            fitness_random_state=fitness_random_state,
            max_depth=max_depth,
        )
    finally:
        (
            _shared.FeatureMaterializer,
            _shared.RandomForestFitness,
            _shared.ResidualEvaluator,
        ) = original_factories


__all__ = [
    "ArchiveProgressTracker",
    "CandidateEvaluator",
    "CandidateGenerator",
    "MaterializingArchiveSearch",
    "MaterializingGeneticProgramming",
    "build_grammar",
    "build_search_algorithm",
    "canonical_expression_key",
]

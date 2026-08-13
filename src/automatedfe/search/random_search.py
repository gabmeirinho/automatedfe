"""Seeded random search over the complete feature grammar."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike

from geneticengine.evaluation.budget import SearchBudget
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.solutions.individual import PhenotypicIndividual

from ..evaluation.fitness import DEFAULT_N_SPLITS
from .search import (
    MaterializingArchiveSearch,
    _build_evaluated_search,
    _SearchComponents,
)


class _RandomCandidateGenerator:
    """Generate one candidate at a time from a seeded representation."""

    exhausted = False

    def __init__(
        self,
        representation: TreeBasedRepresentation,
        random: NativeRandomSource,
    ) -> None:
        self.representation = representation
        self.random = random

    def generate(
        self,
        _previous: Sequence[PhenotypicIndividual],
        _generation: int,
    ) -> list[PhenotypicIndividual]:
        genotype = self.representation.create_genotype(self.random)
        return [
            PhenotypicIndividual(
                genotype=genotype,
                representation=self.representation,
            )
        ]


def build_random_search(
    budget: SearchBudget,
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
    csv_path: str | PathLike[str] | None = None,
    archive_path: str | PathLike[str] | None = None,
) -> MaterializingArchiveSearch:
    """Build the seeded evaluated random-search strategy."""

    def candidate_generator(components: _SearchComponents):
        return _RandomCandidateGenerator(components.representation, components.random)

    return _build_evaluated_search(
        budget,
        candidate_generator_factory=candidate_generator,
        mapping=mapping,
        mmap_dir=mmap_dir,
        feature_cache_dir=feature_cache_dir,
        dataset_path=dataset_path,
        n_splits=n_splits,
        score_metric=score_metric,
        fitness_random_state=fitness_random_state,
        seed=seed,
        max_depth=max_depth,
        csv_path=csv_path,
        archive_path=archive_path,
    )


__all__ = ["build_random_search"]

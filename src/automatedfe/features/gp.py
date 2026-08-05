"""Minimal GP grammar and search configuration.

The complete grammar is::

    Feature -> TransactionCount(window_i)
    window_i -> an integer from 0 to len(WINDOW_CATALOG) - 1

There are no arithmetic operations.  GP only searches for the window of one
transaction-count feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Callable

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import SearchBudget
from geneticengine.grammar import extract_grammar
from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation

from .feature_spec import Aggregation, FeatureSpec, WINDOW_CATALOG, Window


@abstract
class Feature:
    """The grammar's abstract root (a GeneticEngine non-terminal)."""

    def to_feature_spec(self) -> FeatureSpec:
        raise NotImplementedError


@dataclass
class TransactionCount(Feature):
    """The grammar's only feature production."""

    window_i: Annotated[int, IntRange(0, len(WINDOW_CATALOG) - 1)]

    @property
    def window(self) -> Window:
        return WINDOW_CATALOG[self.window_i]

    def to_feature_spec(self) -> FeatureSpec:
        return FeatureSpec(Aggregation.COUNT, None, self.window)

    def __str__(self) -> str:
        return self.to_feature_spec().name


def build_grammar() -> Grammar:
    """Create ``Feature -> TransactionCount(window_i)``."""

    return extract_grammar([TransactionCount], Feature)


def build_search_algorithm(
    grammar: Grammar,
    fitness_function: Callable[[Feature], float],
    budget: SearchBudget,
    *,
    population_size: int = 20,
    seed: int = 42,
) -> GeneticProgramming:
    """Configure GeneticEngine GP with the caller-provided search budget."""

    if population_size <= 0:
        raise ValueError("population_size must be positive")

    random = NativeRandomSource(seed)
    representation = TreeBasedRepresentation(
        grammar,
        MaxDepthDecider(random, grammar, max_depth=1),
    )
    problem = SingleObjectiveProblem(
        fitness_function=fitness_function,
        minimize=False,
    )
    return GeneticProgramming(
        problem=problem,
        budget=budget,
        representation=representation,
        population_size=population_size,
        random=random,
    )


__all__ = [
    "Feature",
    "TransactionCount",
    "build_grammar",
    "build_search_algorithm",
]

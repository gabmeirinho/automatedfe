"""Minimal GP grammar and search configuration.

The complete grammar is::

    Mean(
        feature: one of ["amount"],
        window: 0 .. len(WINDOW_CATALOG) - 1,
    )

There are no arithmetic operations. GP searches for the window used to take
the mean of the one available base feature, ``"amount"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Annotated, Callable

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import SearchBudget
from geneticengine.evaluation.recorder import CSVSearchRecorder
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.vars import VarRange
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation

from .feature_spec import (
    AMOUNT_COLUMN,
    Aggregation,
    FeatureSpec,
    WINDOW_CATALOG,
    Window as FeatureWindow,
)


# Constrained terminal types keep Mean at tree depth 1. Adding another base
# feature later only requires adding its column name to VarRange.
Feature = Annotated[str, VarRange([AMOUNT_COLUMN])]
WindowIndex = Annotated[int, IntRange(0, len(WINDOW_CATALOG) - 1)]


@dataclass
class Mean:
    """Mean aggregation with exactly two parameters: feature and window."""

    feature: Feature
    window: WindowIndex

    @property
    def selected_window(self) -> FeatureWindow:
        return WINDOW_CATALOG[self.window]

    def to_feature_spec(self) -> FeatureSpec:
        return FeatureSpec(Aggregation.MEAN, self.feature, self.selected_window)

    def __str__(self) -> str:
        return self.to_feature_spec().name


def build_grammar() -> Grammar:
    """Create the depth-1 ``Mean(feature, window)`` grammar."""

    return extract_grammar([Mean], Mean)


def build_search_algorithm(
    grammar: Grammar,
    fitness_function: Callable[[Mean], float],
    budget: SearchBudget,
    *,
    population_size: int = 20,
    seed: int = 42,
    csv_path: str | PathLike[str] | None = None,
) -> GeneticProgramming:
    """Configure GP and optionally record every evaluation to a CSV file."""

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

    return GeneticProgramming(
        problem=problem,
        budget=budget,
        representation=representation,
        population_size=population_size,
        random=random,
        tracker=tracker,
    )


__all__ = [
    "Feature",
    "Mean",
    "WindowIndex",
    "build_grammar",
    "build_search_algorithm",
]

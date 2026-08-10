"""Compatibility facade for the relocated enumeration and random searches.

The strategy implementations now live in ``features.search``.  This module is
kept only for callers that still import the historical path.
"""

from __future__ import annotations

from typing import Any

from geneticengine.evaluation.budget import SearchBudget
from geneticengine.grammar.grammar import Grammar

from .feature_materialization import FeatureMaterializer
from .fitness import RandomForestFitness, ResidualEvaluator
from .search import enumerative_search as _enumerative
from .search import random_search as _random
from .search import search as _shared
from .search.enumerative_search import EnumerationResult
from .search.search import (
    ARCHIVE_MINIMIZE,
    DEFAULT_MAX_DEPTH,
    CandidateGenerator,
    _SearchComponents,
    canonical_expression_key,
)

iterate_grammar = _enumerative.iterate_grammar
_EnumerativeCandidateGenerator = _enumerative._EnumerativeCandidateGenerator
_RandomCandidateGenerator = _random._RandomCandidateGenerator
_build_search_components = _shared._build_search_components


class BoundedExpressionEnumerator(_enumerative.BoundedExpressionEnumerator):
    """Legacy wrapper that preserves the old ``iterate_grammar`` patch point."""

    def __init__(
        self,
        grammar: Grammar,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        starting_symbol: type | None = None,
    ) -> None:
        super().__init__(
            grammar,
            max_depth=max_depth,
            starting_symbol=starting_symbol,
            iterator_factory=iterate_grammar,
        )


def iter_bounded_expressions(
    grammar: Grammar,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    starting_symbol: type | None = None,
) -> BoundedExpressionEnumerator:
    return BoundedExpressionEnumerator(
        grammar,
        max_depth=max_depth,
        starting_symbol=starting_symbol,
    )


def collect_unique_expressions(
    grammar: Grammar,
    candidate_count: int,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    starting_symbol: type | None = None,
) -> EnumerationResult:
    if candidate_count <= 0:
        raise ValueError(f"candidate_count must be positive, got {candidate_count}")
    stream = iter_bounded_expressions(
        grammar,
        max_depth=max_depth,
        starting_symbol=starting_symbol,
    )
    expressions = []
    while len(expressions) < candidate_count:
        try:
            expressions.append(next(stream))
        except StopIteration:
            break
    return EnumerationResult(tuple(expressions), stream.exhausted)


def build_enumerative_search(
    budget: SearchBudget,
    **kwargs: Any,
):
    """Build the relocated enumerative strategy."""

    return _build_with_legacy_factories(
        _enumerative.build_enumerative_search,
        budget,
        kwargs,
    )


def build_random_search(
    budget: SearchBudget,
    **kwargs: Any,
):
    """Build the relocated random strategy."""

    return _build_with_legacy_factories(_random.build_random_search, budget, kwargs)


def _build_with_legacy_factories(builder, budget, kwargs):
    original_factories = (
        _shared.FeatureMaterializer,
        _shared.RandomForestFitness,
        _shared.ResidualEvaluator,
    )
    _shared.FeatureMaterializer = FeatureMaterializer
    _shared.RandomForestFitness = RandomForestFitness
    _shared.ResidualEvaluator = ResidualEvaluator
    try:
        return builder(budget, **kwargs)
    finally:
        (
            _shared.FeatureMaterializer,
            _shared.RandomForestFitness,
            _shared.ResidualEvaluator,
        ) = original_factories


BoundedGrammarEnumerator = BoundedExpressionEnumerator
collect_evaluation_free_expressions = collect_unique_expressions


__all__ = [
    "ARCHIVE_MINIMIZE",
    "DEFAULT_MAX_DEPTH",
    "BoundedExpressionEnumerator",
    "BoundedGrammarEnumerator",
    "EnumerationResult",
    "build_enumerative_search",
    "build_random_search",
    "canonical_expression_key",
    "collect_evaluation_free_expressions",
    "collect_unique_expressions",
    "iter_bounded_expressions",
]

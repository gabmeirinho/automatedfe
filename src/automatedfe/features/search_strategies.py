"""Candidate streams shared by the feature-search strategies.

Genetic Engine owns the grammar enumerator used here.  This module only adds
the project-specific concerns around that enumerator: a depth bound, structural
deduplication, and an observable exhaustion result.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from geneticengine.algorithms.enumerative import iterate_grammar
from geneticengine.grammar.grammar import Grammar

from .archive import encode_expression
from .feature_types import TxFeature
from .grammar import build_grammar, expr, tree_depth

DEFAULT_MAX_DEPTH = 4


def canonical_expression_key(expression: object) -> str:
    """Return a stable structural key for an expression.

    Grammar expressions are keyed from the allowlisted archive representation,
    never from their display string.  The small fallback keeps the historical
    archive and final-evaluation APIs usable with non-grammar test doubles and
    legacy feature descriptors; real grammar candidates always take the first
    branch.
    """

    if isinstance(expression, expr):
        encoded: object = encode_expression(expression)
    elif isinstance(expression, TxFeature):
        encoded = {
            "type": "TxFeature",
            "name": expression.name,
        }
    else:
        encoded = {
            "type": f"{type(expression).__module__}.{type(expression).__qualname__}",
            "value": str(expression),
        }
    return json.dumps(encoded, sort_keys=True, separators=(",", ":"))


class BoundedExpressionEnumerator(Iterator[expr]):
    """Iterate unique grammar expressions up to ``max_depth``.

    The order is exactly the first-occurrence order produced by Genetic
    Engine's :func:`iterate_grammar`.  In particular, values above the depth
    bound are skipped rather than treated as end-of-stream: Genetic Engine can
    replay shallower values while expanding a later recursive level.
    """

    def __init__(
        self,
        grammar: Grammar,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        starting_symbol: type | None = None,
    ) -> None:
        if max_depth <= 0:
            raise ValueError(f"max_depth must be positive, got {max_depth}")
        self.grammar = grammar
        self.max_depth = max_depth
        self.starting_symbol = starting_symbol or grammar.starting_symbol
        self._source = iter(iterate_grammar(grammar, self.starting_symbol))
        self._seen: set[str] = set()
        self._exhausted = False

    @property
    def exhausted(self) -> bool:
        """Whether the delegated Genetic Engine stream has ended."""

        return self._exhausted

    @property
    def grammar_exhausted(self) -> bool:
        """Alias used by runner diagnostics for the exhaustion state."""

        return self.exhausted

    @property
    def seen_keys(self) -> frozenset[str]:
        """Return the structural identities emitted by this run so far."""

        return frozenset(self._seen)

    def __iter__(self) -> BoundedExpressionEnumerator:
        return self

    def __next__(self) -> expr:
        while True:
            try:
                candidate = next(self._source)
            except StopIteration:
                self._exhausted = True
                raise

            if not isinstance(candidate, expr):
                raise TypeError(
                    "iterate_grammar yielded a non-expression candidate: "
                    f"{type(candidate).__name__}"
                )
            if tree_depth(candidate) > self.max_depth:
                continue

            key = canonical_expression_key(candidate)
            if key in self._seen:
                continue
            self._seen.add(key)
            return candidate


@dataclass(frozen=True, slots=True)
class EnumerationResult:
    """Expressions collected without materialization or fitness evaluation."""

    expressions: tuple[expr, ...]
    exhausted: bool

    def __len__(self) -> int:
        return len(self.expressions)

    @property
    def grammar_exhausted(self) -> bool:
        """Whether collection consumed the delegated stream to its end."""

        return self.exhausted

    def __iter__(self) -> Iterator[expr]:
        return iter(self.expressions)


def iter_bounded_expressions(
    grammar: Grammar,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    starting_symbol: type | None = None,
) -> BoundedExpressionEnumerator:
    """Create a deterministic, unique, bounded expression stream."""

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
    """Collect at most ``candidate_count`` unique expressions.

    This function only consumes the grammar stream.  It deliberately has no
    materializer, fitness evaluator, or other evaluation hook in its API.
    """

    if candidate_count <= 0:
        raise ValueError(
            f"candidate_count must be positive, got {candidate_count}"
        )
    stream = iter_bounded_expressions(
        grammar,
        max_depth=max_depth,
        starting_symbol=starting_symbol,
    )
    expressions: list[expr] = []
    while len(expressions) < candidate_count:
        try:
            expressions.append(next(stream))
        except StopIteration:
            break
    return EnumerationResult(tuple(expressions), stream.exhausted)


def iter_feature_expressions(
    mapping: Any = None,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> BoundedExpressionEnumerator:
    """Create the bounded stream for the complete feature grammar."""

    return iter_bounded_expressions(
        build_grammar(mapping),
        max_depth=max_depth,
    )


def collect_feature_expressions(
    candidate_count: int,
    mapping: Any = None,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> EnumerationResult:
    """Collect complete-grammar expressions without evaluating them."""

    return collect_unique_expressions(
        build_grammar(mapping),
        candidate_count,
        max_depth=max_depth,
    )


# These aliases make the strategy intent explicit at call sites while keeping
# one implementation of the traversal and collection behavior.
BoundedGrammarEnumerator = BoundedExpressionEnumerator
collect_evaluation_free_expressions = collect_unique_expressions


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "BoundedExpressionEnumerator",
    "BoundedGrammarEnumerator",
    "EnumerationResult",
    "canonical_expression_keyf",
    "collect_evaluation_free_expressions",
    "collect_feature_expressions",
    "collect_unique_expressions",
    "iter_bounded_expressions",
    "iter_feature_expressions",
]

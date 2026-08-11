"""Enumerative candidate generation without evaluation or archive state."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from os import PathLike

from geneticengine.grammar.grammar import Grammar

from ..grammar import build_grammar, expr
from .enumerative_search import (
    DEFAULT_MAX_DEPTH,
    EnumerationResult,
    iter_bounded_expressions,
)
from .search import canonical_expression_key


class UnboundEnumerativeSearch:
    """Generate a bounded batch of expressions without evaluating them."""

    def __init__(
        self,
        *,
        grammar: Grammar,
        max_depth: int,
        candidate_count: int,
    ) -> None:
        self.grammar = grammar
        self.max_depth = max_depth
        self.candidate_count = candidate_count
        self.generated_count = 0
        self.duplicate_count = 0
        self.accepted_count = 0
        self.expressions: tuple[expr, ...] = ()
        self.enumeration_result: EnumerationResult | None = None
        self._seen: set[str] = set()
        self.candidate_observers: list[Callable[[expr], None]] = []

    @property
    def grammar_exhausted(self) -> bool:
        return bool(
            self.enumeration_result is not None
            and self.enumeration_result.exhausted
        )

    @property
    def seen_keys(self) -> frozenset[str]:
        return frozenset(self._seen)

    def search(self) -> list[expr]:
        """Generate and return expressions in deterministic grammar order."""

        stream = iter_bounded_expressions(
            self.grammar,
            max_depth=self.max_depth,
        )
        expressions: list[expr] = []
        while len(expressions) < self.candidate_count:
            try:
                expression = next(stream)
            except StopIteration:
                break
            expressions.append(expression)
            for observer in self.candidate_observers:
                observer(expression)
        self.enumeration_result = EnumerationResult(
            tuple(expressions),
            stream.exhausted,
        )
        self.expressions = self.enumeration_result.expressions
        self.generated_count = len(self.expressions)
        self.accepted_count = len(self.expressions)
        self._seen = {
            canonical_expression_key(expression) for expression in self.expressions
        }
        return list(self.expressions)


def build_unbound_enumerative_search(
    candidate_count: int,
    *,
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    max_depth: int | None = None,
) -> UnboundEnumerativeSearch:
    """Build an unbound generator for one batch of enumerated expressions.

    The returned expressions can be passed directly to
    :class:`automatedfe.features.final_evaluation.FinalEvaluator`.
    """

    if candidate_count <= 0:
        raise ValueError(f"candidate_count must be positive, got {candidate_count}")
    if max_depth is None:
        max_depth = DEFAULT_MAX_DEPTH
    if max_depth <= 0:
        raise ValueError("max_depth must be positive")
    return UnboundEnumerativeSearch(
        grammar=build_grammar(mapping),
        max_depth=max_depth,
        candidate_count=candidate_count,
    )


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "UnboundEnumerativeSearch",
    "build_unbound_enumerative_search",
]

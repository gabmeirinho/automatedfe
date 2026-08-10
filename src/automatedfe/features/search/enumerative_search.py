"""Archive-backed enumerative search and bounded grammar enumeration."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike

from geneticengine.algorithms.enumerative import iterate_grammar
from geneticengine.evaluation.budget import SearchBudget
from geneticengine.grammar.grammar import Grammar
from geneticengine.solutions.individual import ConcreteIndividual, PhenotypicIndividual

from ..fitness import DEFAULT_N_SPLITS
from ..grammar import expr, tree_depth
from .search import (
    DEFAULT_MAX_DEPTH,
    MaterializingArchiveSearch,
    _SearchComponents,
    _build_evaluated_search,
    canonical_expression_key,
)


class BoundedExpressionEnumerator(Iterator[expr]):
    """Iterate unique grammar expressions up to ``max_depth``."""

    def __init__(
        self,
        grammar: Grammar,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        starting_symbol: type | None = None,
        iterator_factory: Callable[[Grammar, type], Iterator[object]] | None = None,
    ) -> None:
        if max_depth <= 0:
            raise ValueError(f"max_depth must be positive, got {max_depth}")
        self.grammar = grammar
        self.max_depth = max_depth
        self.starting_symbol = starting_symbol or grammar.starting_symbol
        source_factory = (
            iterate_grammar if iterator_factory is None else iterator_factory
        )
        self._source = iter(source_factory(grammar, self.starting_symbol))
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
        """Return structural identities emitted by this run so far."""

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
                self._exhausted = True
                raise StopIteration

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
    """Collect at most ``candidate_count`` unique expressions."""

    if candidate_count <= 0:
        raise ValueError(f"candidate_count must be positive, got {candidate_count}")
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


class _EnumerativeCandidateGenerator:
    """Generate one candidate at a time from the bounded grammar stream."""

    def __init__(self, grammar: Grammar, *, max_depth: int) -> None:
        self.stream = iter_bounded_expressions(grammar, max_depth=max_depth)

    @property
    def exhausted(self) -> bool:
        return self.stream.exhausted

    def generate(
        self,
        _previous: Sequence[PhenotypicIndividual],
        _generation: int,
    ) -> list[PhenotypicIndividual]:
        try:
            return [ConcreteIndividual(next(self.stream))]
        except StopIteration:
            return []


def build_enumerative_search(
    budget: SearchBudget,
    *,
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    mmap_dir: str | PathLike[str],
    feature_cache_dir: str | PathLike[str] | None = None,
    dataset_path: str | PathLike[str] | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    score_metric: str = "roc_auc",
    fitness_random_state: int = 42,
    seed: int = 42,
    max_depth: int | None = None,
    csv_path: str | PathLike[str] | None = None,
    archive_path: str | PathLike[str] | None = None,
) -> MaterializingArchiveSearch:
    """Build the evaluated archive-backed enumerative strategy."""

    def candidate_generator(components: _SearchComponents):
        return _EnumerativeCandidateGenerator(
            components.grammar,
            max_depth=components.max_depth,
        )

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


BoundedGrammarEnumerator = BoundedExpressionEnumerator
collect_evaluation_free_expressions = collect_unique_expressions


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "BoundedExpressionEnumerator",
    "BoundedGrammarEnumerator",
    "EnumerationResult",
    "build_enumerative_search",
    "canonical_expression_key",
    "collect_evaluation_free_expressions",
    "collect_unique_expressions",
    "iter_bounded_expressions",
]

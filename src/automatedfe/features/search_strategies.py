"""Candidate generation for the four feature-search strategies.

Genetic Engine owns the grammar enumerator used here.  This module only adds
the project-specific depth bound, structural identity, and generators. Genetic,
enumerative, and random search all delegate evaluation to the same archive
search lifecycle; archive-free enumeration stops after candidate generation.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from geneticengine.algorithms.enumerative import iterate_grammar
from geneticengine.evaluation.budget import SearchBudget
from geneticengine.problems import MultiObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.grammar.grammar import Grammar
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.solutions.individual import (
    ConcreteIndividual,
    PhenotypicIndividual,
)

from .archive import encode_expression
from .archive import ArchiveStep
from .feature_materialization import FeatureMaterializer
from .fitness import (
    DEFAULT_N_SPLITS,
    RandomForestFitness,
    ResidualEvaluator,
)
from .feature_types import TxFeature
from .grammar import build_grammar, expr, tree_depth

if TYPE_CHECKING:
    from .gp import MaterializingArchiveSearch

DEFAULT_MAX_DEPTH = 4
ARCHIVE_MINIMIZE = [False, False, False, True]


@dataclass(frozen=True, slots=True)
class _SearchComponents:
    """Reusable configured objects shared by all evaluated strategies."""

    grammar: Grammar
    representation: TreeBasedRepresentation
    materializer: FeatureMaterializer
    fitness_evaluator: Any
    problem: MultiObjectiveProblem
    archive_step: ArchiveStep
    random: NativeRandomSource
    max_depth: int


def _build_search_components(
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
    archive_path: str | PathLike[str] | None = None,
    materializer_factory: Callable[..., FeatureMaterializer] | None = None,
    random_fitness_factory: Callable[..., Any] | None = None,
    residual_fitness_factory: Callable[..., Any] | None = None,
) -> _SearchComponents:
    """Build the common grammar, evaluation, and archive foundation.

    The factories are intentionally injectable.  Besides making the setup
    easy to test, this preserves the historical ``gp`` module's ability to
    substitute its materializer and fitness evaluator during integration
    tests.
    """

    if dataset_path is None:
        raise ValueError("dataset_path is required for archive search")
    if n_splits != 3:
        raise ValueError("Archive mode requires exactly three folds")
    if materializer_factory is None:
        materializer_factory = FeatureMaterializer
    if random_fitness_factory is None:
        random_fitness_factory = RandomForestFitness
    if residual_fitness_factory is None:
        residual_fitness_factory = ResidualEvaluator
    grammar = build_grammar(mapping)
    if max_depth is None:
        max_depth = DEFAULT_MAX_DEPTH
    if max_depth <= 0:
        raise ValueError("max_depth must be positive")

    materializer = materializer_factory(mmap_dir, features_dir=feature_cache_dir)
    random = NativeRandomSource(seed)
    representation = TreeBasedRepresentation(
        grammar,
        MaxDepthDecider(random, grammar, max_depth=max_depth),
    )
    if score_metric in {"brier", "brier_improvement"}:
        fitness_evaluator = residual_fitness_factory(
            materializer,
            dataset_path,
            n_splits=n_splits,
            score_metric=score_metric,
        )
    else:
        fitness_evaluator = random_fitness_factory(
            materializer,
            dataset_path,
            n_splits=n_splits,
            score_metric=score_metric,
            random_state=fitness_random_state,
        )
    objective_vector = getattr(fitness_evaluator, "objective_vector", None)
    if not callable(objective_vector):
        raise TypeError("Archive search requires an objective_vector evaluator")
    problem = MultiObjectiveProblem(
        fitness_function=objective_vector,
        minimize=list(ARCHIVE_MINIMIZE),
    )
    if archive_path is not None:
        resolved_archive_path = Path(archive_path).resolve()
        if resolved_archive_path.exists() and resolved_archive_path.is_dir():
            raise ValueError(
                "archive_path must identify a file, not a directory: "
                f"{resolved_archive_path}"
            )
    archive_step = ArchiveStep(archive_path=archive_path, mapping=mapping)
    return _SearchComponents(
        grammar=grammar,
        representation=representation,
        materializer=materializer,
        fitness_evaluator=fitness_evaluator,
        problem=problem,
        archive_step=archive_step,
        random=random,
        max_depth=max_depth,
    )


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


def build_enumerative_search(
    budget: SearchBudget,
    **kwargs: Any,
) -> MaterializingArchiveSearch:
    """Build the evaluated bounded enumerative strategy."""

    return _build_generated_search("enumerative", budget, **kwargs)


def build_random_search(
    budget: SearchBudget,
    **kwargs: Any,
) -> MaterializingArchiveSearch:
    """Build the seeded evaluated random-search strategy."""

    return _build_generated_search("random", budget, **kwargs)


def _build_generated_search(
    strategy: str,
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
    """Attach a proposal generator to the common evaluated-search lifecycle."""

    # Local import avoids a module cycle: gp.py uses the component builder
    # above for the genetic strategy.
    from .gp import ArchiveProgressTracker, MaterializingArchiveSearch, _csv_recorder

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
    candidate_generator = (
        _EnumerativeCandidateGenerator(
            components.grammar,
            max_depth=components.max_depth,
        )
        if strategy == "enumerative"
        else _RandomCandidateGenerator(components.representation, components.random)
    )

    recorder = _csv_recorder(csv_path, components.problem)
    tracker = ArchiveProgressTracker(
        components.problem,
        components.archive_step,
        recorders=[] if recorder is None else [recorder],
    )
    return MaterializingArchiveSearch(
        problem=components.problem,
        budget=budget,
        representation=components.representation,
        population_size=1,
        random=components.random,
        tracker=tracker,
        materializer=components.materializer,
        fitness_evaluator=components.fitness_evaluator,
        archive_step=components.archive_step,
        candidate_generator=candidate_generator,
        deduplicate=True,
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
    "build_enumerative_search",
    "build_random_search",
    "canonical_expression_key",
    "collect_evaluation_free_expressions",
    "collect_unique_expressions",
    "iter_bounded_expressions",
]

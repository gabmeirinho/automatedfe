"""Feature-search strategies and their shared evaluated-search lifecycle."""

from .enumerative_search import (
    BoundedExpressionEnumerator,
    BoundedGrammarEnumerator,
    EnumerationResult,
    build_enumerative_search,
    collect_evaluation_free_expressions,
    collect_unique_expressions,
    iter_bounded_expressions,
)
from .gp import MaterializingGeneticProgramming, build_search_algorithm
from .random_search import build_random_search
from .search import (
    ARCHIVE_MINIMIZE,
    DEFAULT_MAX_DEPTH,
    ArchiveProgressTracker,
    CandidateEvaluator,
    CandidateGenerator,
    MaterializingArchiveSearch,
    canonical_expression_key,
)

__all__ = [
    "ARCHIVE_MINIMIZE",
    "DEFAULT_MAX_DEPTH",
    "ArchiveProgressTracker",
    "BoundedExpressionEnumerator",
    "BoundedGrammarEnumerator",
    "CandidateEvaluator",
    "CandidateGenerator",
    "EnumerationResult",
    "MaterializingArchiveSearch",
    "MaterializingGeneticProgramming",
    "build_enumerative_search",
    "build_random_search",
    "build_search_algorithm",
    "canonical_expression_key",
    "collect_evaluation_free_expressions",
    "collect_unique_expressions",
    "iter_bounded_expressions",
]

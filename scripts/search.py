"""Run a feature-search strategy and evaluate its selected features."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from automatedfe.encoding import DEFAULT_MAPPING_OUTPUT
from automatedfe.features import (
    DEFAULT_MAX_DEPTH,
    SearchStrategy,
    SearchRunResult,
    run_feature_search,
    write_summary_json,
)
from automatedfe.fitness import DEFAULT_RANDOM_STATE
from automatedfe.sorting import DEFAULT_DATASET_OUTPUT
from automatedfe.transaction_materialization import DEFAULT_MMAP_DIR


DEFAULT_DATASET = DEFAULT_DATASET_OUTPUT
DEFAULT_MAPPING = DEFAULT_MAPPING_OUTPUT
DEFAULT_SEED = 42
DEFAULT_POPULATION_SIZE = 50

_EVALUATED_STRATEGIES = (
    SearchStrategy.GENETIC,
    SearchStrategy.ENUMERATIVE,
    SearchStrategy.RANDOM,
)


def _positive_int(value: str) -> int:
    try:
        converted = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if converted <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return converted


def build_parser() -> argparse.ArgumentParser:
    """Build the strategy-aware search argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Search the complete feature grammar and evaluate the selected "
            "features on the held-out split."
        )
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=tuple(strategy.value for strategy in SearchStrategy),
        help="Search strategy to run",
    )
    parser.add_argument(
        "--time-budget",
        "--time-budget-seconds",
        type=_positive_int,
        help="Positive integer search time budget in seconds (evaluated strategies only)",
    )
    parser.add_argument(
        "--candidate-count",
        type=_positive_int,
        help=(
            "Number of unique expressions to generate "
            "(enumerative_without_archive only)"
        ),
    )
    parser.add_argument(
        "--dataset",
        "--dataset-path",
        dest="dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Sorted event dataset parquet file (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--mapping",
        "--mapping-path",
        dest="mapping",
        type=Path,
        default=DEFAULT_MAPPING,
        help=f"Label-mapping JSON file (default: {DEFAULT_MAPPING})",
    )
    parser.add_argument(
        "--mmap-dir",
        type=Path,
        default=DEFAULT_MMAP_DIR,
        help=f"Directory containing transaction mmap columns (default: {DEFAULT_MMAP_DIR})",
    )
    parser.add_argument(
        "--feature-cache-dir",
        "--feature-cache",
        type=Path,
        default=None,
        help="Persistent primitive/event feature cache directory (default: disabled)",
    )
    parser.add_argument(
        "--score-metric",
        choices=("accuracy", "roc_auc", "brier", "brier_improvement"),
        default="brier_improvement",
        help="Cross-validation score metric (default: brier_improvement)",
    )
    parser.add_argument(
        "--fitness-random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=(
            "Random state used by the search-time fitness models "
            f"(default: {DEFAULT_RANDOM_STATE})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for candidate generation (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--population-size",
        type=_positive_int,
        default=DEFAULT_POPULATION_SIZE,
        help=f"Genetic population size (default: {DEFAULT_POPULATION_SIZE})",
    )
    parser.add_argument(
        "--max-depth",
        type=_positive_int,
        default=DEFAULT_MAX_DEPTH,
        help=f"Maximum grammar tree depth (default: {DEFAULT_MAX_DEPTH})",
    )
    parser.add_argument(
        "--csv",
        "--csv-path",
        "--diagnostics-csv",
        dest="csv_path",
        type=Path,
        help="Optional common candidate diagnostics CSV path",
    )
    parser.add_argument(
        "--archive",
        "--archive-path",
        "--evaluated-archive",
        dest="archive_path",
        type=Path,
        help="Optional final Pareto archive JSON path (evaluated strategies only)",
    )
    parser.add_argument(
        "--summary",
        "--summary-json",
        "--summary-json-path",
        dest="summary_path",
        type=Path,
        help="Optional atomic run summary JSON path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing CSV, archive, or summary outputs",
    )
    return parser


def _validate_strategy_options(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> SearchStrategy:
    strategy = SearchStrategy(args.strategy)
    evaluated = strategy in _EVALUATED_STRATEGIES
    if evaluated:
        if args.time_budget is None:
            parser.error(
                f"--time-budget is required for strategy {strategy.value!r}"
            )
        if args.candidate_count is not None:
            parser.error(
                f"--candidate-count is only valid for strategy "
                f"{SearchStrategy.ENUMERATIVE_WITHOUT_ARCHIVE.value!r}"
            )
    else:
        if args.candidate_count is None:
            parser.error(
                f"--candidate-count is required for strategy {strategy.value!r}"
            )
        if args.time_budget is not None:
            parser.error(
                f"--time-budget is only valid for evaluated strategies "
                f"({', '.join(item.value for item in _EVALUATED_STRATEGIES)})"
            )
        if args.archive_path is not None:
            parser.error(
                "--archive is not supported for strategy "
                f"{strategy.value!r}"
            )
    return strategy


def _preflight_outputs(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    paths = {
        "CSV": args.csv_path,
        "archive": args.archive_path,
        "summary": args.summary_path,
    }
    resolved: dict[str, Path] = {}
    for label, path in paths.items():
        if path is None:
            continue
        resolved_path = path.resolve()
        if resolved_path.exists() and resolved_path.is_dir():
            parser.error(f"{label} output path is a directory: {resolved_path}")
        resolved[label] = resolved_path

    if len(set(resolved.values())) != len(resolved):
        parser.error("CSV, archive, and summary outputs must be different files")

    if not args.force:
        for label, path in resolved.items():
            if path.exists():
                parser.error(
                    f"Refusing to overwrite existing {label} output without "
                    f"--force: {path}"
                )


def _path_value(path: Path | None) -> str | None:
    return None if path is None else str(path.resolve())


def _summary_document(
    args: argparse.Namespace,
    strategy: SearchStrategy,
    result: SearchRunResult,
) -> dict[str, object]:
    document: dict[str, object] = {
        "strategy": strategy.value,
        "configuration": {
            "time_budget_seconds": args.time_budget,
            "candidate_count": args.candidate_count,
            "dataset_path": _path_value(args.dataset),
            "mapping": _path_value(args.mapping),
            "mmap_dir": _path_value(args.mmap_dir),
            "feature_cache_dir": _path_value(args.feature_cache_dir),
            "score_metric": args.score_metric,
            "fitness_random_state": args.fitness_random_state,
            "seed": args.seed,
            "population_size": args.population_size,
            "max_depth": args.max_depth,
        },
        "counts": {
            "generated": result.generated_count,
            "evaluated": result.evaluated_count,
            "invalid": result.invalid_count,
            "duplicates": result.duplicate_count,
        },
        "timings": {
            "search_seconds": result.search_duration_seconds,
            "final_evaluation_seconds": result.final_evaluation_duration_seconds,
        },
        "grammar_exhausted": result.grammar_exhausted,
        "selected_feature_count": len(result.expressions),
        "final_metrics": dict(result.final_metrics),
    }
    if result.objectives is not None:
        document["objectives"] = [list(objectives) for objectives in result.objectives]
    return document


def _print_result(
    args: argparse.Namespace,
    strategy: SearchStrategy,
    result: SearchRunResult,
) -> None:
    print(f"Strategy: {strategy.value}")
    print(
        "Counts: "
        f"generated={result.generated_count}, "
        f"evaluated={result.evaluated_count}, "
        f"invalid={result.invalid_count}, "
        f"duplicates={result.duplicate_count}"
    )
    print(f"Selected features: {len(result.expressions)}")
    print(
        "Timings: "
        f"search={result.search_duration_seconds:.6f}s, "
        f"final_evaluation={result.final_evaluation_duration_seconds:.6f}s"
    )
    print(f"Grammar exhausted: {result.grammar_exhausted}")
    if result.final_metrics:
        metrics = ", ".join(
            f"{name}={value:.6f}"
            for name, value in sorted(result.final_metrics.items())
        )
        print(f"Final metrics: {metrics}")
    for label, path in (
        ("Diagnostics", args.csv_path),
        ("Archive", args.archive_path),
        ("Summary", args.summary_path),
    ):
        if path is not None:
            print(f"{label}: {path.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the selected strategy, and print its summary."""

    parser = build_parser()
    args = parser.parse_args(argv)
    strategy = _validate_strategy_options(parser, args)
    _preflight_outputs(parser, args)

    try:
        result = run_feature_search(
            strategy,
            time_budget_seconds=args.time_budget,
            candidate_count=args.candidate_count,
            dataset_path=args.dataset,
            mapping=args.mapping,
            mmap_dir=args.mmap_dir,
            feature_cache_dir=args.feature_cache_dir,
            score_metric=args.score_metric,
            fitness_random_state=args.fitness_random_state,
            seed=args.seed,
            population_size=args.population_size,
            max_depth=args.max_depth,
            csv_path=args.csv_path,
            archive_path=args.archive_path,
            force=args.force,
        )
        if args.summary_path is not None:
            write_summary_json(
                args.summary_path,
                _summary_document(args, strategy, result),
                force=args.force,
            )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    _print_result(args, strategy, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_MAPPING",
    "DEFAULT_MMAP_DIR",
    "build_parser",
    "main",
]

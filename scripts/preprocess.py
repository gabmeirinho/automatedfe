"""Command-line interface for the full preprocessing pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from automatedfe.preprocessing import (
    DEFAULT_CARD_TOKENS_INPUT,
    DEFAULT_DATASET_INPUT,
    DEFAULT_DATASET_OUTPUT,
    DEFAULT_INPUT,
    DEFAULT_MAPPING_OUTPUT,
    DEFAULT_MERCHANTS_INPUT,
    DEFAULT_MMAP_DIR,
    DEFAULT_OUTPUT,
    preprocess,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transactions",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Raw transactions parquet file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--transformed",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Sorted, enriched, encoded parquet file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--card-tokens",
        type=Path,
        default=DEFAULT_CARD_TOKENS_INPUT,
        help=f"Card-token parquet file (default: {DEFAULT_CARD_TOKENS_INPUT})",
    )
    parser.add_argument(
        "--merchants",
        type=Path,
        default=DEFAULT_MERCHANTS_INPUT,
        help=f"Merchant parquet file (default: {DEFAULT_MERCHANTS_INPUT})",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_INPUT,
        help=f"Event dataset parquet file (default: {DEFAULT_DATASET_INPUT})",
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=DEFAULT_DATASET_OUTPUT,
        help=f"Sorted event dataset parquet file (default: {DEFAULT_DATASET_OUTPUT})",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_OUTPUT,
        help=f"Label-mapping JSON file (default: {DEFAULT_MAPPING_OUTPUT})",
    )
    parser.add_argument(
        "--mmap-dir",
        type=Path,
        default=DEFAULT_MMAP_DIR,
        help=f"Directory for materialized mmap column files (default: {DEFAULT_MMAP_DIR})",
    )
    parser.add_argument(
        "--no-materialize",
        action="store_true",
        help="Skip materializing the encoded transactions into mmap files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing sort and materialization outputs",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a live DuckDB progress bar for long-running queries",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    preprocess(
        args.transactions,
        args.transformed,
        card_tokens_path=args.card_tokens,
        merchants_path=args.merchants,
        dataset_path=args.dataset,
        dataset_output_path=args.dataset_output,
        mapping_path=args.mapping,
        mmap_dir=args.mmap_dir,
        materialize=not args.no_materialize,
        force=args.force,
        progress=args.progress,
    )
    print(f"Encoded transactions written to {args.transformed.resolve()}")
    print(f"Label mapping written to {args.mapping.resolve()}")
    if not args.no_materialize:
        print(f"Mmap columns written to {args.mmap_dir.resolve()}")


if __name__ == "__main__":
    main()

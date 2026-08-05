"""Command-line interface for the full preprocessing pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from automatedfe.preprocessing import (
    DEFAULT_CARD_TOKENS_INPUT,
    DEFAULT_DATASET_INPUT,
    DEFAULT_DATASET_OUTPUT,
    DEFAULT_INPUT,
    DEFAULT_MAPPING_OUTPUT,
    DEFAULT_MERCHANTS_INPUT,
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
        "--force",
        action="store_true",
        help="Replace existing sort outputs",
    )
    args = parser.parse_args()

    preprocess(
        args.transactions,
        args.transformed,
        card_tokens_path=args.card_tokens,
        merchants_path=args.merchants,
        dataset_path=args.dataset,
        dataset_output_path=args.dataset_output,
        mapping_path=args.mapping,
        force=args.force,
    )
    print(f"Encoded transactions written to {args.transformed.resolve()}")
    print(f"Label mapping written to {args.mapping.resolve()}")


if __name__ == "__main__":
    main()

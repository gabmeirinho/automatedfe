"""Command-line interface for sorting transactions."""

from __future__ import annotations

import argparse
from pathlib import Path

from automatedfe.sorting import (
    DEFAULT_CARD_TOKENS_INPUT,
    DEFAULT_INPUT,
    DEFAULT_MERCHANTS_INPUT,
    DEFAULT_OUTPUT,
    sort_transactions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input parquet file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Sorted parquet file (default: {DEFAULT_OUTPUT})",
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
        "--force",
        action="store_true",
        help="Replace the output file if it already exists",
    )
    args = parser.parse_args()
    sort_transactions(
        args.input,
        args.output,
        card_tokens_path=args.card_tokens,
        merchants_path=args.merchants,
        force=args.force,
    )
    print(f"Sorted transactions written to {args.output.resolve()}")


if __name__ == "__main__":
    main()

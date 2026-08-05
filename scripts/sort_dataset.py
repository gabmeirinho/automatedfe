"""Command-line interface for sorting the event dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from automatedfe.sorting import (
    DEFAULT_DATASET_INPUT,
    DEFAULT_DATASET_OUTPUT,
    sort_dataset,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATASET_INPUT,
        help=f"Input parquet file (default: {DEFAULT_DATASET_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_OUTPUT,
        help=f"Sorted parquet file (default: {DEFAULT_DATASET_OUTPUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output file if it already exists",
    )
    args = parser.parse_args()
    sort_dataset(args.input, args.output, force=args.force)
    print(f"Sorted dataset written to {args.output.resolve()}")


if __name__ == "__main__":
    main()

"""Command-line interface for sorting transactions."""

from __future__ import annotations

import argparse
from pathlib import Path

from automatedfe.sorting import (
    DEFAULT_INPUT,
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
        "--force",
        action="store_true",
        help="Replace the output file if it already exists",
    )
    args = parser.parse_args()
    sort_transactions(args.input, args.output, force=args.force)
    print(f"Sorted transactions written to {args.output.resolve()}")


if __name__ == "__main__":
    main()

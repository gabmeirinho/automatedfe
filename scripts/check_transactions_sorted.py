"""Command-line interface for checking transaction ordering."""

from __future__ import annotations

import argparse
from pathlib import Path

from automatedfe.data.validation import DEFAULT_OUTPUT, first_sorting_violation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Parquet file to check (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input parquet file does not exist: {args.input}")

    violation = first_sorting_violation(args.input)
    if violation is None:
        print(f"PASS: {args.input} is sorted by merchant_id, created_at")
        return

    position, previous_merchant, merchant, previous_created, created = violation
    print(f"FAIL: {args.input} is not sorted")
    print(f"First violation at row {position}:")
    print(f"  previous: merchant_id={previous_merchant}, created_at={previous_created}")
    print(f"  current:  merchant_id={merchant}, created_at={created}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()

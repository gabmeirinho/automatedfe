"""Command-line interface for label-encoding transactions."""

from __future__ import annotations

import argparse
from pathlib import Path

from automatedfe.encoding import (
    DEFAULT_DATASET_INPUT,
    DEFAULT_INPUT,
    DEFAULT_MAPPING_OUTPUT,
    DEFAULT_OUTPUT,
    encode_transactions,
    fit_label_mapping,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit a label mapping on the train split")
    fit_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input parquet file (default: {DEFAULT_INPUT})",
    )
    fit_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_INPUT,
        help=f"Event dataset parquet file (default: {DEFAULT_DATASET_INPUT})",
    )
    fit_parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_OUTPUT,
        help=f"Output label-mapping JSON file (default: {DEFAULT_MAPPING_OUTPUT})",
    )
    fit_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the mapping file if it already exists",
    )

    encode_parser = subparsers.add_parser(
        "encode", help="Apply a fitted label mapping to transactions"
    )
    encode_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input parquet file (default: {DEFAULT_INPUT})",
    )
    encode_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Encoded parquet file (default: {DEFAULT_OUTPUT})",
    )
    encode_parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_OUTPUT,
        help=f"Label-mapping JSON file (default: {DEFAULT_MAPPING_OUTPUT})",
    )
    encode_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output file if it already exists",
    )

    args = parser.parse_args()
    if args.command == "fit":
        fit_label_mapping(
            args.input,
            args.dataset,
            args.mapping,
            force=args.force,
        )
        print(f"Label mapping written to {args.mapping.resolve()}")
    else:
        encode_transactions(
            args.input,
            args.output,
            args.mapping,
            force=args.force,
        )
        print(f"Encoded transactions written to {args.output.resolve()}")


if __name__ == "__main__":
    main()

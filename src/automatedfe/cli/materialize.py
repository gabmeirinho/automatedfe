"""Command-line interface for materializing transformed transactions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import logging
from pathlib import Path

from automatedfe.data.sorting import DEFAULT_OUTPUT
from automatedfe.data.transaction_materialization import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MMAP_DIR,
    materialize_transactions,
)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Encoded transactions parquet file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MMAP_DIR,
        help=f"Directory for the mmap column files (default: {DEFAULT_MMAP_DIR})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows streamed into the mmap per write (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing materialization",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Log per-chunk and per-column progress",
    )
    args = parser.parse_args(argv)
    materialize_transactions(
        args.input,
        args.output_dir,
        chunk_size=args.chunk_size,
        force=args.force,
        progress=args.progress,
    )
    print(f"Materialized columns written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

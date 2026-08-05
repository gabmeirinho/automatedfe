"""Check mmap row counts against the transformed transactions parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np

from automatedfe.materialization import DEFAULT_MMAP_DIR, MANIFEST_FILENAME
from automatedfe.sorting import DEFAULT_OUTPUT


def parquet_row_count(input_path: Path) -> int:
    """Return the number of rows in a parquet file."""

    connection = duckdb.connect()
    try:
        return connection.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(input_path.resolve())]
        ).fetchone()[0]
    finally:
        connection.close()


def mmap_lengths(mmap_dir: Path) -> dict[str, int]:
    """Return actual element counts for the mmaps described by *mmap_dir*.

    The element count is calculated from each file's byte size and its dtype,
    rather than trusting the row count recorded in the manifest.
    """

    manifest_path = mmap_dir / MANIFEST_FILENAME
    with manifest_path.open() as manifest_file:
        manifest = json.load(manifest_file)

    if not isinstance(manifest, dict):
        raise ValueError(f"Malformed manifest: {manifest_path}")

    columns = manifest.get("columns")
    if not isinstance(columns, dict):
        raise ValueError(f"Malformed manifest: {manifest_path}")

    lengths: dict[str, int] = {}
    for column, metadata in columns.items():
        if not isinstance(metadata, dict) or not isinstance(metadata.get("file"), str):
            raise ValueError(f"Malformed manifest entry for column {column!r}")
        if not isinstance(metadata.get("dtype"), str):
            raise ValueError(f"Malformed manifest entry for column {column!r}")

        mmap_path = mmap_dir / metadata["file"]
        if not mmap_path.is_file():
            raise FileNotFoundError(f"Mmap file does not exist: {mmap_path}")

        dtype = np.dtype(metadata["dtype"])
        if dtype.itemsize <= 0:
            raise ValueError(f"Mmap dtype has no storage width: {mmap_path}")
        if mmap_path.stat().st_size % dtype.itemsize:
            raise ValueError(
                f"Mmap file size is not a multiple of its dtype size: {mmap_path}"
            )
        lengths[column] = mmap_path.stat().st_size // dtype.itemsize

    manifest_files = {metadata["file"] for metadata in columns.values()}
    unexpected = sorted(
        path.name
        for path in mmap_dir.glob("*.mmap")
        if path.name not in manifest_files
    )
    if unexpected:
        raise ValueError(
            "Mmap file(s) are not listed in the manifest: " + ", ".join(unexpected)
        )

    return lengths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transformed",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Transformed transactions parquet (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--mmap-dir",
        type=Path,
        default=DEFAULT_MMAP_DIR,
        help=f"Directory containing mmap files (default: {DEFAULT_MMAP_DIR})",
    )
    args = parser.parse_args()

    if not args.transformed.is_file():
        parser.error(
            f"Transformed transactions parquet does not exist: {args.transformed}"
        )
    if not args.mmap_dir.is_dir():
        parser.error(f"Mmap directory does not exist: {args.mmap_dir}")
    if not (args.mmap_dir / MANIFEST_FILENAME).is_file():
        parser.error(
            f"Mmap manifest does not exist: {args.mmap_dir / MANIFEST_FILENAME}"
        )

    try:
        expected_rows = parquet_row_count(args.transformed)
        lengths = mmap_lengths(args.mmap_dir)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    mismatches = {
        column: length for column, length in lengths.items() if length != expected_rows
    }

    print(f"Transformed transactions: {expected_rows} rows")
    for column, length in lengths.items():
        status = "OK" if length == expected_rows else "MISMATCH"
        print(f"{column}.mmap: {length} rows ({status})")

    if mismatches:
        print("FAIL: mmap lengths do not match the transformed transactions dataset")
        return 1

    print(f"PASS: all {len(lengths)} mmap files contain {expected_rows} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

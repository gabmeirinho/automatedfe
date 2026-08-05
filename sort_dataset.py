"""Sort the dataset parquet file by its event timestamp.

By default, the sorted data is written to
``data/loan/transformed/dataset_sorted.parquet`` so the source file is preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "loan" / "dataset.parquet"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "loan" / "transformed" / "dataset_sorted.parquet"
)


def sort_dataset(input_path: Path, output_path: Path, *, force: bool = False) -> None:
    """Sort *input_path* by ``event_timestamp`` into *output_path*."""

    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet file does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("The output must be a different file from the input.")
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path}. Re-run with --force to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    try:
        columns = {
            row[0]
            for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(input_path)]
            ).fetchall()
        }
        if "event_timestamp" not in columns:
            raise ValueError(
                "Input parquet is missing the required column: event_timestamp"
            )

        # The output path is resolved above and escaped before being inserted
        # into the COPY statement; the input path remains a bound parameter.
        output_sql_path = str(output_path).replace("'", "''")
        connection.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet(?)
                ORDER BY event_timestamp
            )
            TO '{output_sql_path}'
            (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
            """,
            [str(input_path)],
        )
    finally:
        connection.close()

    print(f"Sorted dataset written to {output_path}")


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
    sort_dataset(args.input, args.output, force=args.force)


if __name__ == "__main__":
    main()

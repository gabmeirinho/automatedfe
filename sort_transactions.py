"""Sort the transactions parquet dataset with DuckDB.

The source dataset calls its timestamp column ``created_at``. By default the
sorted data is written to ``data/loan/transformed/transactions.parquet`` so
the original file is preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "loan" / "transactions.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "loan" / "transformed" / "transactions.parquet"
DUCKDB_MEMORY_LIMIT = "16GB"


def sort_transactions(input_path: Path, output_path: Path, *, force: bool = False) -> None:
    """Sort *input_path* by merchant and creation time into *output_path*."""

    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet file does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError(
            "The output must be a different file from the input. "
            "Use a separate sorted output and replace the source manually if needed."
        )
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path}. Re-run with --force to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    try:
        # Keep the sort from consuming all available system memory. DuckDB
        # will spill intermediate sort data to its temporary directory when
        # necessary.
        connection.execute("SET memory_limit = ?", [DUCKDB_MEMORY_LIMIT])

        # read_parquet's parameter keeps paths safe without interpolating them
        # into SQL.  The explicit column check gives a useful error if a
        # different parquet file is passed accidentally.
        columns = {
            row[0]
            for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(input_path)]
            ).fetchall()
        }
        required_columns = {"merchant_id", "created_at"}
        missing_columns = required_columns - columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Input parquet is missing required column(s): {missing}")

        # DuckDB does not bind a parameter used as the COPY destination in the
        # same way as a read_parquet parameter.  Quote the already-resolved
        # filesystem path explicitly, escaping the only SQL-significant
        # character in a path.
        output_sql_path = str(output_path).replace("'", "''")
        connection.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet(?)
                ORDER BY merchant_id, created_at
            )
            TO '{output_sql_path}'
            (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
            """,
            [str(input_path)],
        )
    finally:
        connection.close()

    print(f"Sorted transactions written to {output_path}")


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


if __name__ == "__main__":
    main()

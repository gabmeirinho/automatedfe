"""Compare null counts and percentages for source and transformed transactions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import duckdb


from automatedfe.data.sorting import PROJECT_ROOT


DEFAULT_TRANSACTIONS_INPUT = PROJECT_ROOT / "data" / "loan" / "transactions.parquet"
DEFAULT_TRANSFORMED_INPUT = (
    PROJECT_ROOT / "data" / "loan" / "transformed" / "transactions.parquet"
)


def quote_identifier(identifier: str) -> str:
    """Quote a DuckDB identifier safely."""

    return '"' + identifier.replace('"', '""') + '"'


def null_statistics(
    connection: duckdb.DuckDBPyConnection, input_path: Path
) -> tuple[int, list[tuple[str, int]]]:
    """Return the row count and null count for every column in a parquet file."""

    schema = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(input_path)]
    ).fetchall()
    expressions = [
        f'count(*) - count({quote_identifier(column[0])})'
        for column in schema
    ]
    result = connection.execute(
        f"""
        SELECT count(*) AS total_rows, {', '.join(expressions)}
        FROM read_parquet(?)
        """,
        [str(input_path)],
    ).fetchone()

    return result[0], [
        (column[0], result[index]) for index, column in enumerate(schema, start=1)
    ]


def format_statistic(total_rows: int, null_count: int) -> tuple[str, str, str]:
    """Format row count, null count, and null percentage for output."""

    null_percent = 100.0 * null_count / total_rows if total_rows else 0.0
    return str(total_rows), str(null_count), f"{null_percent:.6f}%"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transactions",
        type=Path,
        default=DEFAULT_TRANSACTIONS_INPUT,
        help=f"Source transactions parquet (default: {DEFAULT_TRANSACTIONS_INPUT})",
    )
    parser.add_argument(
        "--transformed",
        type=Path,
        default=DEFAULT_TRANSFORMED_INPUT,
        help=f"Transformed transactions parquet (default: {DEFAULT_TRANSFORMED_INPUT})",
    )
    args = parser.parse_args(argv)

    for path, label in (
        (args.transactions, "Transactions"),
        (args.transformed, "Transformed transactions"),
    ):
        if not path.exists():
            parser.error(f"{label} parquet file does not exist: {path}")

    connection = duckdb.connect()
    try:
        original_rows, original_statistics = null_statistics(
            connection, args.transactions.resolve()
        )
        transformed_rows, transformed_statistics = null_statistics(
            connection, args.transformed.resolve()
        )

        original_by_column = dict(original_statistics)
        transformed_by_column = dict(transformed_statistics)
        columns = list(original_by_column)
        columns.extend(
            column for column in transformed_by_column if column not in original_by_column
        )

        print(
            "column\toriginal_rows\toriginal_null_count\toriginal_null_percent\t"
            "transformed_rows\ttransformed_null_count\ttransformed_null_percent"
        )
        for column in columns:
            original = (
                format_statistic(original_rows, original_by_column[column])
                if column in original_by_column
                else ("N/A", "N/A", "N/A")
            )
            transformed = (
                format_statistic(transformed_rows, transformed_by_column[column])
                if column in transformed_by_column
                else ("N/A", "N/A", "N/A")
            )
            print(f"{column}\t{'\t'.join(original)}\t{'\t'.join(transformed)}")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

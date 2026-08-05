"""Sort the project Parquet datasets."""

from __future__ import annotations

from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "loan" / "transactions.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "loan" / "transformed" / "transactions.parquet"
DUCKDB_MEMORY_LIMIT = "16GB"
DEFAULT_DATASET_INPUT = PROJECT_ROOT / "data" / "loan" / "dataset.parquet"
DEFAULT_DATASET_OUTPUT = (
    PROJECT_ROOT / "data" / "loan" / "transformed" / "dataset_sorted.parquet"
)


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
        # same way as a read_parquet parameter. The resolved path is escaped
        # before it is inserted into the COPY statement.
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

"""Sort the project Parquet datasets."""

from __future__ import annotations

from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "loan" / "transactions.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "loan" / "transformed" / "transactions.parquet"
DEFAULT_CARD_TOKENS_INPUT = PROJECT_ROOT / "data" / "loan" / "card_tokens.parquet"
DEFAULT_MERCHANTS_INPUT = PROJECT_ROOT / "data" / "loan" / "merchants.parquet"
DUCKDB_MEMORY_LIMIT = "16GB"
DEFAULT_DATASET_INPUT = PROJECT_ROOT / "data" / "loan" / "dataset.parquet"
DEFAULT_DATASET_OUTPUT = (
    PROJECT_ROOT / "data" / "loan" / "transformed" / "dataset.parquet"
)


def _columns(connection: duckdb.DuckDBPyConnection, input_path: Path) -> set[str]:
    """Return the columns in a parquet file."""

    return {
        row[0]
        for row in connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(input_path)]
        ).fetchall()
    }


def _validate_input_path(input_path: Path, label: str) -> Path:
    input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"{label} parquet file does not exist: {input_path}")
    return input_path


def sort_transactions(
    input_path: Path,
    output_path: Path,
    *,
    card_tokens_path: Path | None = None,
    merchants_path: Path | None = None,
    force: bool = False,
) -> None:
    """Enrich and sort transactions by merchant and creation time.

    When relation paths are supplied, ``card_tokens.card_brand`` and
    ``merchants.document_type`` are left-joined onto the transaction rows.
    Leaving either path as ``None`` skips that enrichment.
    """

    input_path = _validate_input_path(input_path, "Input")
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

        columns = _columns(connection, input_path)
        required_columns = {"merchant_id", "created_at"}
        missing_columns = required_columns - columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Input parquet is missing required column(s): {missing}")

        relation_paths: list[tuple[str, Path, str]] = []
        if card_tokens_path is not None:
            card_tokens_path = _validate_input_path(card_tokens_path, "Card tokens")
            card_token_columns = _columns(connection, card_tokens_path)
            missing = {"id", "card_brand"} - card_token_columns
            if missing:
                missing_names = ", ".join(sorted(missing))
                raise ValueError(
                    f"Card-token parquet is missing required column(s): {missing_names}"
                )
            if "card_token_id" not in columns:
                raise ValueError(
                    "Input parquet is missing the required column: card_token_id"
                )
            relation_paths.append(("card_tokens", card_tokens_path, "card_brand"))

        if merchants_path is not None:
            merchants_path = _validate_input_path(merchants_path, "Merchants")
            merchant_columns = _columns(connection, merchants_path)
            missing = {"id", "document_type"} - merchant_columns
            if missing:
                missing_names = ", ".join(sorted(missing))
                raise ValueError(
                    f"Merchant parquet is missing required column(s): {missing_names}"
                )
            relation_paths.append(("merchants", merchants_path, "document_type"))

        # If an already-enriched input is passed in, replace the enrichment
        # columns instead of producing duplicate names in the output schema.
        enrichment_columns = [output_column for _, _, output_column in relation_paths]
        excluded_columns = [column for column in enrichment_columns if column in columns]
        transaction_projection = "t.*"
        if excluded_columns:
            quoted = ", ".join(f'"{column}"' for column in excluded_columns)
            transaction_projection = f"t.* EXCLUDE ({quoted})"

        select_columns = [transaction_projection]
        joins = []
        parameters = [str(input_path)]
        for relation_name, relation_path, output_column in relation_paths:
            alias = "ct" if relation_name == "card_tokens" else "m"
            join_key = "card_token_id" if alias == "ct" else "merchant_id"
            select_columns.append(f"{alias}.{output_column} AS {output_column}")
            joins.append(
                f"LEFT JOIN read_parquet(?) AS {alias} ON t.{join_key} = {alias}.id"
            )
            parameters.append(str(relation_path))

        # DuckDB does not bind a parameter used as the COPY destination in the
        # same way as a read_parquet parameter. The resolved path is escaped
        # before it is inserted into the COPY statement.
        output_sql_path = str(output_path).replace("'", "''")
        connection.execute(
            f"""
            COPY (
                SELECT {', '.join(select_columns)}
                FROM read_parquet(?) AS t
                {' '.join(joins)}
                ORDER BY t.merchant_id, t.created_at
            )
            TO '{output_sql_path}'
            (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
            """,
            parameters,
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

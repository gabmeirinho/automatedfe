"""Sort the project Parquet datasets."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)


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


def _schema(
    connection: duckdb.DuckDBPyConnection, input_path: Path
) -> list[tuple[str, str]]:
    """Return the columns and DuckDB types in a parquet file."""

    return [
        (row[0], row[1])
        for row in connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(input_path)]
        ).fetchall()
    ]


def _zero_filled_expression(
    column: str, column_type: str, *, table_alias: str = "t"
) -> str:
    """Build a type-preserving expression that replaces NULL with zero."""

    quoted_column = f'{table_alias}."{column}"'
    normalized_type = column_type.upper()

    if normalized_type.startswith("VARCHAR") or normalized_type in {
        "CHAR",
        "TEXT",
    }:
        default = "'0'"
    elif normalized_type.startswith("TIMESTAMP"):
        default = (
            "TIMESTAMPTZ '1970-01-01 00:00:00+00:00'"
            if "WITH TIME ZONE" in normalized_type
            else "TIMESTAMP '1970-01-01 00:00:00'"
        )
    elif normalized_type == "DATE":
        default = "DATE '1970-01-01'"
    elif normalized_type.startswith("TIME"):
        default = "TIME '00:00:00'"
    else:
        # DuckDB can infer the correct zero type for numeric and boolean
        # columns from the first argument to COALESCE.
        default = "0"

    return f"coalesce({quoted_column}, {default}) AS \"{column}\""


def _validate_input_path(input_path: Path, label: str) -> Path:
    input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"{label} parquet file does not exist: {input_path}")
    return input_path


def _enable_progress_bar(connection: duckdb.DuckDBPyConnection) -> None:
    """Show a live progress bar for queries running longer than 500 ms."""

    connection.execute("PRAGMA enable_progress_bar")
    connection.execute("PRAGMA progress_bar_time = 500")
    logger.info("Progress bar enabled (shows after 0.5s of query time)")


def sort_transactions(
    input_path: Path,
    output_path: Path,
    *,
    card_tokens_path: Path | None = None,
    merchants_path: Path | None = None,
    force: bool = False,
    progress: bool = False,
) -> None:
    """Enrich and sort transactions by merchant and creation time.

    When relation paths are supplied, ``card_tokens.card_brand`` and
    ``merchants.document_type`` are left-joined onto the transaction rows.
    If both inputs contain ``merchant_category_code``, null transaction codes
    are filled from the matching merchant while non-null transaction codes are
    preserved. If neither source has a code, ``"0"`` is used as the resolved
    value in the single ``merchant_category_code`` output column. Leaving
    either path as ``None`` skips that enrichment.

    When *progress* is true, DuckDB prints a live progress bar to stderr for
    queries that run longer than 0.5 seconds.
    """

    input_path = _validate_input_path(input_path, "Input")
    output_path = output_path.resolve()

    logger.info("Sorting transactions: %s", input_path)
    logger.info("Writing sorted transactions to: %s", output_path)
    logger.info("Using a %s DuckDB memory limit", DUCKDB_MEMORY_LIMIT)

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
    if output_path.exists() and force:
        logger.info("Replacing existing output: %s", output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    connection = duckdb.connect()
    try:
        # Keep the sort from consuming all available system memory. DuckDB
        # will spill intermediate sort data to its temporary directory when
        # necessary.
        connection.execute("SET memory_limit = ?", [DUCKDB_MEMORY_LIMIT])
        if progress:
            _enable_progress_bar(connection)

        input_schema = _schema(connection, input_path)
        columns = {column for column, _ in input_schema}
        required_columns = {"merchant_id", "created_at"}
        missing_columns = required_columns - columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Input parquet is missing required column(s): {missing}")
        logger.info(
            "Validated input schema: %d columns, sorted by merchant_id, created_at",
            len(columns),
        )

        relation_paths: list[tuple[str, Path, str]] = []
        resolve_merchant_category_code = False
        if card_tokens_path is not None:
            card_tokens_path = _validate_input_path(card_tokens_path, "Card tokens")
            logger.info("Enriching with card_brand from: %s", card_tokens_path)
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
            logger.info("Enriching with document_type from: %s", merchants_path)
            merchant_columns = _columns(connection, merchants_path)
            missing = {"id", "document_type"} - merchant_columns
            if missing:
                missing_names = ", ".join(sorted(missing))
                raise ValueError(
                    f"Merchant parquet is missing required column(s): {missing_names}"
                )
            if "merchant_category_code" in columns:
                if "merchant_category_code" not in merchant_columns:
                    raise ValueError(
                        "Merchant parquet is missing the required column: "
                        "merchant_category_code"
                    )
                resolve_merchant_category_code = True
                logger.info(
                    "Resolving null merchant_category_code values from merchants"
                )
            relation_paths.append(("merchants", merchants_path, "document_type"))

        # If an already-enriched input is passed in, replace the enrichment
        # columns instead of producing duplicate names in the output schema.
        enrichment_columns = [output_column for _, _, output_column in relation_paths]
        excluded_columns = {column for column in enrichment_columns if column in columns}
        if resolve_merchant_category_code:
            excluded_columns.add("merchant_category_code")
        select_columns = [
            _zero_filled_expression(column, column_type)
            for column, column_type in input_schema
            if column not in excluded_columns
        ]
        joins = []
        parameters = [str(input_path)]
        for relation_name, relation_path, output_column in relation_paths:
            alias = "ct" if relation_name == "card_tokens" else "m"
            join_key = "card_token_id" if alias == "ct" else "merchant_id"
            relation_schema = _schema(connection, relation_path)
            relation_type = dict(relation_schema)[output_column]
            select_columns.append(
                _zero_filled_expression(
                    output_column, relation_type, table_alias=alias
                )
            )
            joins.append(
                f"LEFT JOIN read_parquet(?) AS {alias} ON t.{join_key} = {alias}.id"
            )
            parameters.append(str(relation_path))

        if resolve_merchant_category_code:
            select_columns.append(
                'coalesce(t."merchant_category_code", '
                'm."merchant_category_code", \'0\') AS "merchant_category_code"'
            )

        # DuckDB does not bind a parameter used as the COPY destination in the
        # same way as a read_parquet parameter. The resolved path is escaped
        # before it is inserted into the COPY statement.
        output_sql_path = str(output_path).replace("'", "''")
        logger.info(
            "Reading, enriching, and sorting %d columns by merchant_id, created_at",
            len(select_columns),
        )
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

    elapsed = time.perf_counter() - start
    logger.info(
        "Sorted transactions written to %s (%.1fs)",
        output_path,
        elapsed,
    )


def sort_dataset(
    input_path: Path, output_path: Path, *, force: bool = False, progress: bool = False
) -> None:
    """Sort *input_path* by ``event_timestamp`` into *output_path*.

    When *progress* is true, DuckDB prints a live progress bar to stderr for
    queries that run longer than 0.5 seconds.
    """

    input_path = input_path.resolve()
    output_path = output_path.resolve()

    logger.info("Sorting dataset: %s", input_path)
    logger.info("Writing sorted dataset to: %s", output_path)
    logger.info("Using a %s DuckDB memory limit", DUCKDB_MEMORY_LIMIT)

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet file does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("The output must be a different file from the input.")
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path}. Re-run with --force to replace it."
        )
    if output_path.exists() and force:
        logger.info("Replacing existing output: %s", output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    connection = duckdb.connect()
    try:
        if progress:
            _enable_progress_bar(connection)
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
        logger.info(
            "Validated input schema: %d columns, sorted by event_timestamp",
            len(columns),
        )

        output_sql_path = str(output_path).replace("'", "''")
        logger.info("Sorting %d columns by event_timestamp", len(columns))
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

    elapsed = time.perf_counter() - start
    logger.info("Sorted dataset written to %s (%.1fs)", output_path, elapsed)

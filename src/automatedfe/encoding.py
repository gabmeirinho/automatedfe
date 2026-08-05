"""Label-encode the categorical columns of the sorted transaction data.

The label mapping is fitted on the training split only (joined from the event
dataset via ``merchant_id``) and persisted as JSON so the same encoding can be
reused at evaluation time without leaking information from unseen data.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import duckdb

from .sorting import (
    DEFAULT_DATASET_INPUT,
    DEFAULT_OUTPUT as DEFAULT_INPUT,
    DUCKDB_MEMORY_LIMIT,
    PROJECT_ROOT,
)

DEFAULT_OUTPUT = DEFAULT_INPUT
DEFAULT_MAPPING_OUTPUT = (
    PROJECT_ROOT / "data" / "loan" / "transformed" / "label_mapping.json"
)
TRAIN_SPLIT = "train"

CATEGORICAL_COLUMNS = (
    "status",
    "capture_method",
    "payment_method",
    "card_brand",
    "document_type",
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


def _validate_input_path(input_path: Path, label: str) -> Path:
    input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"{label} parquet file does not exist: {input_path}")
    return input_path


def _require_columns(
    connection: duckdb.DuckDBPyConnection,
    input_path: Path,
    required: set[str],
    label: str,
) -> None:
    missing = required - _columns(connection, input_path)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"{label} parquet is missing required column(s): {missing_names}")


def _sql_literal(value: object) -> str:
    """Return a SQL string literal for a mapping category."""

    return "'" + str(value).replace("'", "''") + "'"


def _encoded_expression(column: str, mapping: dict[str, int]) -> str:
    """Build a CASE expression that maps *column* values to labels.

    Categories not present in the mapping are assigned a reserved label equal
    to the number of known categories (``max_label + 1``).
    """

    unknown = len(mapping)
    cases = " ".join(
        f"WHEN {_sql_literal(category)} THEN {label}"
        for category, label in sorted(mapping.items(), key=lambda item: item[1])
    )
    clause = f"CASE t.\"{column}\" {cases}" if cases else f"CASE t.\"{column}\""
    return f"{clause} ELSE {unknown} END AS \"{column}\""


def load_label_mapping(mapping_path: Path) -> dict[str, dict[str, int]]:
    """Load a label mapping persisted by :func:`fit_label_mapping`."""

    mapping_path = mapping_path.resolve()
    with open(mapping_path) as mapping_file:
        return json.load(mapping_file)


def fit_label_mapping(
    input_path: Path,
    dataset_path: Path,
    mapping_path: Path,
    *,
    columns: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, dict[str, int]]:
    """Fit a label mapping on the training split and persist it as JSON.

    Distinct category values are collected per column from transaction rows
    whose ``merchant_id`` belongs to the ``train`` split of *dataset_path*.
    Labels are assigned in lexicographic category order so that the mapping is
    reproducible across runs. *columns* defaults to :data:`CATEGORICAL_COLUMNS`.
    """

    input_path = _validate_input_path(input_path, "Input")
    dataset_path = _validate_input_path(dataset_path, "Dataset")
    mapping_path = mapping_path.resolve()

    if mapping_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {mapping_path}. Re-run with --force to replace it."
        )
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    columns = list(columns or CATEGORICAL_COLUMNS)

    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit = ?", [DUCKDB_MEMORY_LIMIT])
        _require_columns(connection, input_path, set(columns) | {"merchant_id"}, "Input")
        _require_columns(connection, dataset_path, {"merchant_id", "split"}, "Dataset")

        mapping: dict[str, dict[str, int]] = {}
        for column in columns:
            rows = connection.execute(
                f"""
                SELECT DISTINCT t."{column}"
                FROM read_parquet(?) AS t
                JOIN read_parquet(?) AS d
                  ON t.merchant_id = d.merchant_id
                WHERE d.split = ?
                  AND t."{column}" IS NOT NULL
                ORDER BY 1
                """,
                [str(input_path), str(dataset_path), TRAIN_SPLIT],
            ).fetchall()
            mapping[column] = {row: label for label, (row,) in enumerate(rows)}
    finally:
        connection.close()

    with open(mapping_path, "w") as mapping_file:
        json.dump(mapping, mapping_file, indent=2)

    return mapping


def encode_transactions(
    input_path: Path,
    output_path: Path,
    mapping: dict[str, dict[str, int]] | Path,
    *,
    force: bool = False,
) -> None:
    """Label-encode the categorical columns of *input_path* into *output_path*.

    *mapping* is either a dict (as returned by :func:`fit_label_mapping`) or a
    path to a persisted mapping. Non-categorical columns are copied unchanged.

    When *output_path* equals *input_path* the file is rewritten in place:
    the encoded result is first written to a temporary sibling file and then
    atomically replaces the original. Overwriting an existing file (including
    an in-place encode of the transformed dataset) requires ``force=True``.
    """

    input_path = _validate_input_path(input_path, "Input")
    output_path = output_path.resolve()

    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path}. Re-run with --force to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(mapping, Path):
        mapping = load_label_mapping(mapping)

    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit = ?", [DUCKDB_MEMORY_LIMIT])
        input_schema = _schema(connection, input_path)
        input_columns = {column for column, _ in input_schema}
        missing = set(mapping) - input_columns
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(
                f"Input parquet is missing encoded column(s): {missing_names}"
            )

        select_columns = [
            _encoded_expression(column, mapping[column]) if column in mapping else f't."{column}"'
            for column, _ in input_schema
        ]

        write_path = output_path
        if input_path == output_path:
            write_path = output_path.with_suffix(output_path.suffix + ".tmp")
        output_sql_path = str(write_path).replace("'", "''")
        connection.execute(
            f"""
            COPY (
                SELECT {", ".join(select_columns)}
                FROM read_parquet(?) AS t
            )
            TO '{output_sql_path}'
            (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
            """,
            [str(input_path)],
        )
    finally:
        connection.close()

    if input_path == output_path:
        os.replace(write_path, output_path)


__all__ = [
    "CATEGORICAL_COLUMNS",
    "DEFAULT_DATASET_INPUT",
    "DEFAULT_INPUT",
    "DEFAULT_MAPPING_OUTPUT",
    "DEFAULT_OUTPUT",
    "DUCKDB_MEMORY_LIMIT",
    "PROJECT_ROOT",
    "TRAIN_SPLIT",
    "encode_transactions",
    "fit_label_mapping",
    "load_label_mapping",
]

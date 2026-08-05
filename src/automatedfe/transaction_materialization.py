"""Materialize the transformed transactions into memory-mapped column files.

Each materializable column of the encoded transactions parquet is streamed
from DuckDB in bounded batches and written into its own raw binary file backed
by a ``numpy.memmap``. Writing in batches keeps peak memory at roughly one
batch per column instead of a full column; reading later through the memmap is
lazy, because the OS pages the file in on demand.

Only numeric and label-encoded columns are materialized (string columns such
as ``merchant_category_code`` are skipped). Timestamps are stored as ``int64``
microseconds since the epoch so window aggregations can compare them directly.

The parquet rows are sorted by ``merchant_id, created_at`` before this step, so
every mmap file preserves that ordering.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from .sorting import DUCKDB_MEMORY_LIMIT, PROJECT_ROOT

logger = logging.getLogger(__name__)


DEFAULT_MMAP_DIR = PROJECT_ROOT / "data" / "loan" / "transformed" / "mmap"
MANIFEST_FILENAME = "manifest.json"
MMAP_SUFFIX = ".mmap"
DEFAULT_CHUNK_SIZE = 5_000_000

_NUMERIC_DTYPES: dict[str, np.dtype] = {
    "TINYINT": np.dtype(np.int8),
    "SMALLINT": np.dtype(np.int16),
    "INTEGER": np.dtype(np.int32),
    "BIGINT": np.dtype(np.int64),
    "UTINYINT": np.dtype(np.uint8),
    "USMALLINT": np.dtype(np.uint16),
    "UINTEGER": np.dtype(np.uint32),
    "UBIGINT": np.dtype(np.uint64),
    "FLOAT": np.dtype(np.float32),
    "REAL": np.dtype(np.float32),
    "DOUBLE": np.dtype(np.float64),
}


def _validate_input_path(input_path: Path, label: str) -> Path:
    input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"{label} parquet file does not exist: {input_path}")
    return input_path


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


def column_dtype(column_type: str) -> np.dtype | None:
    """Return the numpy dtype to store a DuckDB column type in, or None.

    None marks a column as not materializable: strings, booleans, decimals and
    other non-numeric types are skipped. Timestamps and dates are stored as
    epoch integer scalars so windowed aggregations stay purely numeric.
    """

    normalized = column_type.upper()
    dtype = _NUMERIC_DTYPES.get(normalized)
    if dtype is not None:
        return dtype
    if normalized.startswith("TIMESTAMP"):
        return np.dtype(np.int64)
    if normalized == "DATE":
        return np.dtype(np.int32)
    if normalized.startswith("DECIMAL"):
        return np.dtype(np.float64)
    return None


def _select_expression(column: str, column_type: str) -> str:
    """Build a SELECT expression converting *column* to its storage scalar."""

    normalized = column_type.upper()
    if normalized.startswith("TIMESTAMP"):
        return f'epoch_us(t."{column}") AS "{column}"'
    if normalized == "DATE":
        return f'epoch_days(t."{column}") AS "{column}"'
    if normalized.startswith("DECIMAL"):
        return f'CAST(t."{column}" AS DOUBLE) AS "{column}"'
    return f't."{column}"'


def _flush_batches(
    arrays: list[pa.Array], rows: int, memmap: np.memmap, offset: int
) -> int:
    """Write *arrays* (``rows`` total rows) into *memmap* at *offset*."""

    combined = pa.concat_arrays(arrays)
    if combined.null_count:
        combined = combined.fill_null(0)
    memmap[offset : offset + rows] = combined.to_numpy(zero_copy_only=False)
    return offset + rows


def _materialize_column(
    connection: duckdb.DuckDBPyConnection,
    input_path: Path,
    column: str,
    column_type: str,
    dtype: np.dtype,
    rows: int,
    memmap_path: Path,
    *,
    chunk_size: int,
    progress: bool,
) -> None:
    """Stream *column* into a preallocated memory-mapped file at *memmap_path*.

    The memmap is flushed and closed even if the stream fails partway.
    """

    memmap = np.memmap(memmap_path, dtype=dtype, mode="w+", shape=(rows,))
    try:
        reader = connection.execute(
            f"SELECT {_select_expression(column, column_type)} "
            "FROM read_parquet(?) AS t",
            [str(input_path)],
        ).to_arrow_reader()

        pending: list[pa.Array] = []
        pending_rows = 0
        offset = 0
        for batch in reader:
            pending.append(batch.column(0))
            pending_rows += batch.num_rows
            if pending_rows >= chunk_size:
                offset = _flush_batches(pending, pending_rows, memmap, offset)
                pending, pending_rows = [], 0
                if progress:
                    logger.info("    %s: wrote %d/%d rows", column, offset, rows)
        if pending_rows:
            offset = _flush_batches(pending, pending_rows, memmap, offset)
            if progress:
                logger.info("    %s: wrote %d/%d rows", column, offset, rows)

        if offset != rows:
            raise ValueError(
                f"Column {column!r} produced {offset} rows, expected {rows}"
            )
    finally:
        memmap.flush()
        memmap._mmap.close()


def read_manifest(output_dir: Path) -> dict[str, object]:
    """Read the manifest describing a materialized column directory."""

    manifest_path = Path(output_dir).resolve() / MANIFEST_FILENAME
    with open(manifest_path) as manifest_file:
        return json.load(manifest_file)


def load_mmapped_columns(
    output_dir: Path = DEFAULT_MMAP_DIR,
    columns: list[str] | None = None,
) -> dict[str, np.memmap]:
    """Open the materialized columns as read-only memory-mapped arrays.

    Returns a mapping of column name to an ``np.memmap`` of shape
    ``(rows,)``. Passing *columns* loads a subset; unknown names raise a
    ``ValueError``.
    """

    output_dir = output_dir.resolve()
    manifest = read_manifest(output_dir)
    manifest_columns = manifest["columns"]
    if not isinstance(manifest_columns, dict):
        raise ValueError(f"Malformed manifest: {output_dir / MANIFEST_FILENAME}")
    names = list(manifest_columns) if columns is None else list(columns)

    missing = [name for name in names if name not in manifest_columns]
    if missing:
        raise ValueError(f"Column(s) not materialized: {', '.join(sorted(missing))}")

    rows = manifest["rows"]
    if not isinstance(rows, int):
        raise ValueError(f"Malformed manifest: {output_dir / MANIFEST_FILENAME}")

    loaded: dict[str, np.memmap] = {}
    for name in names:
        info = manifest_columns[name]
        loaded[name] = np.memmap(
            output_dir / info["file"],
            dtype=np.dtype(info["dtype"]),
            mode="r",
            shape=(rows,),
        )
    return loaded


def materialize_transactions(
    input_path: Path,
    output_dir: Path = DEFAULT_MMAP_DIR,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    force: bool = False,
    progress: bool = False,
) -> None:
    """Materialize each numeric column of *input_path* into a memory-mapped file.

    Columns are written one at a time: DuckDB streams each column in bounded
    RecordBatches and *chunk_size* rows are accumulated before being written
    into a preallocated ``np.memmap``, so peak memory stays independent of the
    dataset size. A ``manifest.json`` recording the row count and per-column
    dtype is written alongside the column files.

    When *progress* is true, per-chunk and per-column progress is logged.
    Overwriting an existing materialization requires ``force=True``.
    """

    input_path = _validate_input_path(input_path, "Input")
    output_dir = output_dir.resolve()
    manifest_path = output_dir / MANIFEST_FILENAME

    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"Materialization already exists: {output_dir}. "
            "Re-run with --force to replace it."
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    start = time.perf_counter()
    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit = ?", [DUCKDB_MEMORY_LIMIT])

        schema = _schema(connection, input_path)
        materializable = [
            (column, column_type, column_dtype(column_type))
            for column, column_type in schema
        ]
        materializable = [
            (column, column_type, dtype)
            for column, column_type, dtype in materializable
            if dtype is not None
        ]
        if not materializable:
            raise ValueError(
                "Input parquet has no materializable (numeric or timestamp) columns"
            )

        rows = connection.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(input_path)]
        ).fetchone()[0]
        logger.info(
            "Materializing %d of %d columns (%d rows) into %s",
            len(materializable),
            len(schema),
            rows,
            output_dir,
        )

        skipped = sorted(
            {column for column, _ in schema} - {column for column, _, _ in materializable}
        )
        if skipped:
            logger.info("Skipping non-numeric column(s): %s", ", ".join(skipped))

        if force and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        columns_manifest: dict[str, dict[str, str]] = {}
        for column, column_type, dtype in materializable:
            column_start = time.perf_counter()
            memmap_path = output_dir / f"{column}{MMAP_SUFFIX}"
            _materialize_column(
                connection,
                input_path,
                column,
                column_type,
                dtype,
                rows,
                memmap_path,
                chunk_size=chunk_size,
                progress=progress,
            )

            columns_manifest[column] = {
                "file": f"{column}{MMAP_SUFFIX}",
                "dtype": dtype.name,
            }
            elapsed = time.perf_counter() - column_start
            logger.info(
                "Materialized %s (%s, %d rows) -> %s (%.1fs)",
                column,
                dtype.name,
                rows,
                memmap_path,
                elapsed,
            )

        manifest = {"rows": rows, "columns": columns_manifest}
        with open(manifest_path, "w") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
    finally:
        connection.close()

    elapsed = time.perf_counter() - start
    logger.info(
        "Materialized %d columns to %s (%.1fs)",
        len(columns_manifest),
        output_dir,
        elapsed,
    )


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_MMAP_DIR",
    "MANIFEST_FILENAME",
    "MMAP_SUFFIX",
    "column_dtype",
    "load_mmapped_columns",
    "materialize_transactions",
    "read_manifest",
]

"""Check that transactions are physically ordered by merchant and timestamp."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from sort_transactions import DEFAULT_OUTPUT, DUCKDB_MEMORY_LIMIT


def first_sorting_violation(input_path: Path) -> tuple | None:
    """Return the first out-of-order pair, or ``None`` when sorted."""

    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit = ?", [DUCKDB_MEMORY_LIMIT])
        # A single thread makes row_number() follow the parquet file's stored
        # order rather than interleaving parallel scan results.
        connection.execute("PRAGMA threads=1")
        return connection.execute(
            """
            WITH ordered AS MATERIALIZED (
                SELECT
                    merchant_id,
                    created_at,
                    row_number() OVER () AS position
                FROM read_parquet(?)
            ), with_previous AS (
                SELECT
                    position,
                    merchant_id,
                    created_at,
                    lag(merchant_id) OVER (ORDER BY position) AS previous_merchant_id,
                    lag(created_at) OVER (ORDER BY position) AS previous_created_at
                FROM ordered
            )
            SELECT
                position,
                previous_merchant_id,
                merchant_id,
                CAST(previous_created_at AS VARCHAR) AS previous_created_at,
                CAST(created_at AS VARCHAR) AS created_at
            FROM with_previous
            WHERE position > 1
              AND (
                    -- DuckDB sorts NULLs last for ascending ORDER BY.
                    (previous_merchant_id IS NULL AND merchant_id IS NOT NULL)
                    OR (
                        previous_merchant_id IS NOT NULL
                        AND merchant_id IS NOT NULL
                        AND merchant_id < previous_merchant_id
                    )
                    OR (
                        merchant_id IS NOT DISTINCT FROM previous_merchant_id
                        AND (
                            (previous_created_at IS NULL AND created_at IS NOT NULL)
                            OR (
                                previous_created_at IS NOT NULL
                                AND created_at IS NOT NULL
                                AND created_at < previous_created_at
                            )
                        )
                    )
                )
            ORDER BY position
            LIMIT 1
            """,
            [str(input_path.resolve())],
        ).fetchone()
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Parquet file to check (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input parquet file does not exist: {args.input}")

    violation = first_sorting_violation(args.input)
    if violation is None:
        print(f"PASS: {args.input} is sorted by merchant_id, created_at")
        return

    position, previous_merchant, merchant, previous_created, created = violation
    print(f"FAIL: {args.input} is not sorted")
    print(f"First violation at row {position}:")
    print(f"  previous: merchant_id={previous_merchant}, created_at={previous_created}")
    print(f"  current:  merchant_id={merchant}, created_at={created}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()

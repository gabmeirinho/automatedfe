"""Print transaction merchant codes that differ from merchants.parquet."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import duckdb


from automatedfe.data.sorting import PROJECT_ROOT


DEFAULT_TRANSACTIONS_INPUT = PROJECT_ROOT / "data" / "loan" / "transactions.parquet"
DEFAULT_MERCHANTS_INPUT = PROJECT_ROOT / "data" / "loan" / "merchants.parquet"


def columns(connection: duckdb.DuckDBPyConnection, input_path: Path) -> set[str]:
    """Return the columns in a parquet file."""

    return {
        row[0]
        for row in connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(input_path)]
        ).fetchall()
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transactions",
        type=Path,
        default=DEFAULT_TRANSACTIONS_INPUT,
        help=f"Transactions parquet file (default: {DEFAULT_TRANSACTIONS_INPUT})",
    )
    parser.add_argument(
        "--merchants",
        type=Path,
        default=DEFAULT_MERCHANTS_INPUT,
        help=f"Merchants parquet file (default: {DEFAULT_MERCHANTS_INPUT})",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print one summarized row per mismatching merchant",
    )
    args = parser.parse_args(argv)

    for path, label in (
        (args.transactions, "Transactions"),
        (args.merchants, "Merchants"),
    ):
        if not path.exists():
            parser.error(f"{label} parquet file does not exist: {path}")

    connection = duckdb.connect()
    try:
        transaction_columns = columns(connection, args.transactions)
        missing_columns = {"merchant_id", "merchant_category_code"} - transaction_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            parser.error(
                f"Transactions parquet is missing required column(s): {missing}"
            )

        merchant_columns = columns(connection, args.merchants)
        missing_columns = {"id", "merchant_category_code"} - merchant_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            parser.error(f"Merchants parquet is missing required column(s): {missing}")

        if args.summary:
            query = """
                WITH transaction_codes AS (
                    SELECT
                        merchant_id,
                        merchant_category_code,
                        count(*) AS transaction_count
                    FROM read_parquet(?)
                    WHERE merchant_id IS NOT NULL
                    GROUP BY merchant_id, merchant_category_code
                )
                SELECT
                    t.merchant_id,
                    m.merchant_category_code AS merchant_category_code,
                    string_agg(
                        COALESCE(t.merchant_category_code, 'NULL'),
                        ', ' ORDER BY t.merchant_category_code NULLS FIRST
                    ) AS transaction_codes,
                    sum(t.transaction_count) AS mismatch_transaction_count,
                    sum(
                        CASE
                            WHEN t.merchant_category_code IS NULL
                            THEN t.transaction_count
                            ELSE 0
                        END
                    ) AS null_transaction_count
                FROM transaction_codes AS t
                INNER JOIN read_parquet(?) AS m ON t.merchant_id = m.id
                -- IS DISTINCT FROM deliberately treats NULL versus a value as a mismatch.
                WHERE t.merchant_category_code IS DISTINCT FROM m.merchant_category_code
                GROUP BY t.merchant_id, m.merchant_category_code
                ORDER BY t.merchant_id
                """
        else:
            query = """
                WITH transaction_codes AS (
                    SELECT
                        merchant_id,
                        merchant_category_code,
                        count(*) AS transaction_count
                    FROM read_parquet(?)
                    WHERE merchant_id IS NOT NULL
                    GROUP BY merchant_id, merchant_category_code
                )
                SELECT
                    t.merchant_id,
                    t.merchant_category_code AS transaction_merchant_category_code,
                    m.merchant_category_code AS merchant_merchant_category_code,
                    t.transaction_count
                FROM transaction_codes AS t
                INNER JOIN read_parquet(?) AS m ON t.merchant_id = m.id
                -- IS DISTINCT FROM deliberately treats NULL versus a value as a mismatch.
                WHERE t.merchant_category_code IS DISTINCT FROM m.merchant_category_code
                ORDER BY t.merchant_id, t.merchant_category_code NULLS FIRST
                """

        rows = connection.execute(
            query,
            [str(args.transactions.resolve()), str(args.merchants.resolve())],
        ).fetchall()
    finally:
        connection.close()

    if not rows:
        print("No merchant category code mismatches found.")
        return 0

    if args.summary:
        print(
            "merchant_id\tmerchant_category_code\ttransaction_codes\t"
            "mismatch_transaction_count\tnull_transaction_count"
        )
        for (
            merchant_id,
            merchant_code,
            transaction_codes,
            mismatch_count,
            null_count,
        ) in rows:
            merchant_code = "NULL" if merchant_code is None else str(merchant_code)
            print(
                f"{merchant_id}\t{merchant_code}\t{transaction_codes}\t"
                f"{mismatch_count}\t{null_count}"
            )
        return 0

    print(
        "merchant_id\ttransaction_merchant_category_code\t"
        "merchant_merchant_category_code\ttransaction_count"
    )
    for merchant_id, transaction_code, merchant_code, transaction_count in rows:
        transaction_code = (
            "NULL" if transaction_code is None else str(transaction_code)
        )
        merchant_code = "NULL" if merchant_code is None else str(merchant_code)
        print(
            f"{merchant_id}\t{transaction_code}\t{merchant_code}\t"
            f"{transaction_count}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

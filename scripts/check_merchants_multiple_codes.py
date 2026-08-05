"""Print merchants that have more than one merchant category code."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "loan" / "merchants.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Merchants parquet file (default: {DEFAULT_INPUT})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input parquet file does not exist: {args.input}")

    connection = duckdb.connect()
    try:
        columns = {
            row[0]
            for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(args.input)]
            ).fetchall()
        }
        required_columns = {"id", "merchant_category_code"}
        missing_columns = required_columns - columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            parser.error(f"Input parquet is missing required column(s): {missing}")

        rows = connection.execute(
            """
            SELECT
                id AS merchant_id,
                count(DISTINCT merchant_category_code) AS merchant_code_count,
                string_agg(
                    DISTINCT merchant_category_code,
                    ', ' ORDER BY merchant_category_code
                ) AS merchant_codes
            FROM read_parquet(?)
            GROUP BY id
            HAVING count(DISTINCT merchant_category_code) > 1
            ORDER BY merchant_id
            """,
            [str(args.input.resolve())],
        ).fetchall()
    finally:
        connection.close()

    if not rows:
        print("No merchants have more than one merchant_category_code.")
        return

    print("merchant_id\tmerchant_code_count\tmerchant_category_codes")
    for merchant_id, code_count, merchant_codes in rows:
        print(f"{merchant_id}\t{code_count}\t{merchant_codes}")


if __name__ == "__main__":
    main()

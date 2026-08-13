import subprocess
import sys

import duckdb


COMMAND = [
    "-m",
    "automatedfe.cli",
    "validate",
    "merchants-code-mismatches",
]


def test_checker_reports_value_and_null_mismatches(tmp_path):
    transactions_path = tmp_path / "transactions.parquet"
    merchants_path = tmp_path / "merchants.parquet"

    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (1, '5812'),
                (1, NULL),
                (1, '5812'),
                (2, NULL),
                (3, '5999')
            AS t(merchant_id, merchant_category_code)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(transactions_path)],
    )
    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (1, '5812'),
                (2, '5812'),
                (3, NULL)
            AS t(id, merchant_category_code)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(merchants_path)],
    )

    result = subprocess.run(
        [
            sys.executable,
            *COMMAND,
            "--transactions",
            str(transactions_path),
            "--merchants",
            str(merchants_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "merchant_id\ttransaction_merchant_category_code\t"
        "merchant_merchant_category_code\ttransaction_count",
        "1\tNULL\t5812\t1",
        "2\tNULL\t5812\t1",
        "3\t5999\tNULL\t1",
    ]

    summary_result = subprocess.run(
        [
            sys.executable,
            *COMMAND,
            "--transactions",
            str(transactions_path),
            "--merchants",
            str(merchants_path),
            "--summary",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert summary_result.stdout.splitlines() == [
        "merchant_id\tmerchant_category_code\ttransaction_codes\t"
        "mismatch_transaction_count\tnull_transaction_count",
        "1\t5812\tNULL\t1\t1",
        "2\t5812\tNULL\t1\t1",
        "3\tNULL\t5999\t1\t0",
    ]

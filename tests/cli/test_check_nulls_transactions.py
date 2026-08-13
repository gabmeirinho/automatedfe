import subprocess
import sys

import duckdb


COMMAND = [
    "-m",
    "automatedfe.cli",
    "validate",
    "nulls-transactions",
]


def test_checker_reports_null_counts_for_both_datasets(tmp_path):
    transactions_path = tmp_path / "transactions.parquet"
    transformed_path = tmp_path / "transformed.parquet"

    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (1, '5812', NULL),
                (2, NULL, 'visa')
            AS t(merchant_id, merchant_category_code, card_brand)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(transactions_path)],
    )
    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (1, '5812', 'visa', 'cnpj'),
                (2, '0', '0', 'cpf')
            AS t(merchant_id, merchant_category_code, card_brand, document_type)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(transformed_path)],
    )

    result = subprocess.run(
        [
            sys.executable,
            *COMMAND,
            "--transactions",
            str(transactions_path),
            "--transformed",
            str(transformed_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "column\toriginal_rows\toriginal_null_count\toriginal_null_percent\t"
        "transformed_rows\ttransformed_null_count\ttransformed_null_percent",
        "merchant_id\t2\t0\t0.000000%\t2\t0\t0.000000%",
        "merchant_category_code\t2\t1\t50.000000%\t2\t0\t0.000000%",
        "card_brand\t2\t1\t50.000000%\t2\t0\t0.000000%",
        "document_type\tN/A\tN/A\tN/A\t2\t0\t0.000000%",
    ]

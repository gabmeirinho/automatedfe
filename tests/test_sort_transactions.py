from pathlib import Path

import duckdb
import pytest

from automatedfe.sorting import sort_transactions
from tests.conftest import write_transactions_fixture


def read_rows(path: Path):
    return duckdb.sql(
        """
        SELECT merchant_id, CAST(created_at AS VARCHAR), description, amount
        FROM read_parquet(?)
        """,
        params=[str(path)],
    ).fetchall()


def test_sort_transactions_orders_rows_and_preserves_data(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    write_transactions_fixture(input_path)

    sort_transactions(input_path, output_path)

    assert read_rows(output_path) == [
        (1, "2024-01-01 00:00:00", "early", 10.0),
        (1, "2024-01-03 00:00:00", "late", 30.0),
        (2, "2024-01-01 00:00:00", "first merchant", 15.0),
        (2, "2024-01-02 00:00:00", "second merchant", 20.0),
    ]


def test_sort_transactions_rejects_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        sort_transactions(tmp_path / "missing.parquet", tmp_path / "output.parquet")


def test_sort_transactions_rejects_same_input_and_output(tmp_path):
    input_path = tmp_path / "input.parquet"
    write_transactions_fixture(input_path)

    with pytest.raises(ValueError, match="different file"):
        sort_transactions(input_path, input_path)


def test_sort_transactions_requires_force_for_existing_output(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    write_transactions_fixture(input_path)
    output_path.touch()

    with pytest.raises(FileExistsError, match="already exists"):
        sort_transactions(input_path, output_path)


def test_sort_transactions_force_replaces_existing_output(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    write_transactions_fixture(input_path)
    output_path.write_bytes(b"not parquet")

    sort_transactions(input_path, output_path, force=True)

    assert len(read_rows(output_path)) == 4


def test_sort_transactions_rejects_missing_required_column(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    duckdb.sql(
        "COPY (SELECT 1 AS merchant_id) TO ? (FORMAT PARQUET)",
        params=[str(input_path)],
    )

    with pytest.raises(ValueError, match="created_at"):
        sort_transactions(input_path, output_path)


def test_sort_transactions_enriches_card_brand_and_document_type(tmp_path):
    input_path = tmp_path / "input.parquet"
    card_tokens_path = tmp_path / "card_tokens.parquet"
    merchants_path = tmp_path / "merchants.parquet"
    output_path = tmp_path / "output.parquet"

    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (20, TIMESTAMP '2024-01-02 00:00:00', 200, '5812'),
                (10, TIMESTAMP '2024-01-02 00:00:00', 100, '5999'),
                (10, TIMESTAMP '2024-01-01 00:00:00', NULL, NULL),
                (30, TIMESTAMP '2024-01-03 00:00:00', 300, NULL)
            AS t(merchant_id, created_at, card_token_id, merchant_category_code)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(input_path)],
    )
    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (100, 'visa'),
                (200, 'mastercard')
            AS t(id, card_brand)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(card_tokens_path)],
    )
    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (10, 'cnpj', '5812'),
                (20, 'cpf', '5812')
            AS t(id, document_type, merchant_category_code)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(merchants_path)],
    )

    sort_transactions(
        input_path,
        output_path,
        card_tokens_path=card_tokens_path,
        merchants_path=merchants_path,
    )

    columns = [
        row[0]
        for row in duckdb.sql(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            params=[str(output_path)],
        ).fetchall()
    ]
    assert "merchant_category_code" in columns

    rows = duckdb.sql(
        """
        SELECT
            merchant_id,
            card_token_id,
            card_brand,
            document_type,
            merchant_category_code
        FROM read_parquet(?)
        """,
        params=[str(output_path)],
    ).fetchall()

    assert rows == [
        (10, 0, "0", "cnpj", "5812"),
        (10, 100, "visa", "cnpj", "5999"),
        (20, 200, "mastercard", "cpf", "5812"),
        (30, 300, "0", "0", "0"),
    ]


def test_sort_transactions_replaces_null_values_with_zero(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"

    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (
                    1,
                    TIMESTAMP '2024-01-01 00:00:00',
                    NULL::BIGINT,
                    NULL::DOUBLE,
                    NULL::VARCHAR
                )
            AS t(merchant_id, created_at, card_token_id, amount, status)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(input_path)],
    )

    sort_transactions(input_path, output_path)

    rows = duckdb.sql(
        """
        SELECT merchant_id, card_token_id, amount, status
        FROM read_parquet(?)
        """,
        params=[str(output_path)],
    ).fetchall()

    assert rows == [(1, 0, 0, "0")]


def test_sort_transactions_replaces_null_mcc_without_merchants_source(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"

    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (1, TIMESTAMP '2024-01-01 00:00:00', NULL::VARCHAR)
            AS t(merchant_id, created_at, merchant_category_code)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(input_path)],
    )

    sort_transactions(input_path, output_path)

    rows = duckdb.sql(
        "SELECT merchant_category_code FROM read_parquet(?)",
        params=[str(output_path)],
    ).fetchall()

    assert rows == [("0",)]

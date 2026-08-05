from pathlib import Path

import duckdb
import pytest

from sort_transactions import sort_transactions
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

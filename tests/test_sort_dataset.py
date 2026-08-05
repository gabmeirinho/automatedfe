from pathlib import Path

import duckdb
import pytest

from sort_dataset import sort_dataset
from tests.conftest import write_dataset_fixture


def read_rows(path: Path):
    return duckdb.sql(
        """
        SELECT CAST(event_timestamp AS VARCHAR), event
        FROM read_parquet(?)
        """,
        params=[str(path)],
    ).fetchall()


def test_sort_dataset_orders_rows(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    write_dataset_fixture(input_path)

    sort_dataset(input_path, output_path)

    assert read_rows(output_path) == [
        ("2024-01-01 00:00:00", "earliest"),
        ("2024-02-01 00:00:00", "middle"),
        ("2024-02-02 00:00:00", "later"),
    ]


def test_sort_dataset_rejects_missing_required_column(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    duckdb.sql(
        "COPY (SELECT 1 AS value) TO ? (FORMAT PARQUET)",
        params=[str(input_path)],
    )

    with pytest.raises(ValueError, match="event_timestamp"):
        sort_dataset(input_path, output_path)


def test_sort_dataset_requires_force_for_existing_output(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    write_dataset_fixture(input_path)
    output_path.touch()

    with pytest.raises(FileExistsError, match="already exists"):
        sort_dataset(input_path, output_path)

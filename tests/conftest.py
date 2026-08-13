from datetime import datetime

import duckdb
import polars as pl


def write_transactions_fixture(path):
    """Write deliberately unsorted transaction data to *path*."""

    pl.DataFrame(
        {
            "merchant_id": [2, 1, 1, 2],
            "created_at": [
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
            ],
            "description": ["second merchant", "late", "early", "first merchant"],
            "amount": [20.0, 30.0, 10.0, 15.0],
        }
    ).write_parquet(path)


def write_dataset_fixture(path):
    """Write deliberately unsorted event data to *path*."""

    pl.DataFrame(
        {
            "event_timestamp": [
                datetime(2024, 2, 2),
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
            ],
            "event": ["later", "earliest", "middle"],
        }
    ).write_parquet(path)


def write_transformed_fixture(tmp_path):
    """Write a parquet mimicking the encoded transactions schema."""

    path = tmp_path / "transformed.parquet"
    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (CAST(1 AS BIGINT), CAST(100.5 AS DOUBLE), 0, TIMESTAMPTZ '2024-01-01 00:00:00+00:00', CAST(7 AS BIGINT), '5812'),
                (CAST(1 AS BIGINT), CAST(200.0 AS DOUBLE), 1, TIMESTAMPTZ '2024-01-02 00:00:00+00:00', CAST(7 AS BIGINT), '5812'),
                (CAST(2 AS BIGINT), CAST(50.25 AS DOUBLE), 0, TIMESTAMPTZ '2024-02-01 00:00:00+00:00', CAST(9 AS BIGINT), '5411'),
                (CAST(2 AS BIGINT), CAST(75.0 AS DOUBLE), 2, TIMESTAMPTZ '2024-02-05 00:00:00+00:00', CAST(9 AS BIGINT), '5411'),
                (CAST(3 AS BIGINT), CAST(10.0 AS DOUBLE), 1, TIMESTAMPTZ '2024-03-01 00:00:00+00:00', CAST(11 AS BIGINT), '0')
            AS t(
                merchant_id,
                amount,
                status,
                created_at,
                card_token_id,
                merchant_category_code
            )
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(path)],
    )
    return path

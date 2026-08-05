from datetime import datetime

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

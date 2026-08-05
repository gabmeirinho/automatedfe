import duckdb

from automatedfe.validation import first_sorting_violation
from tests.conftest import write_transactions_fixture


def test_checker_accepts_sorted_transactions(tmp_path):
    path = tmp_path / "sorted.parquet"
    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (1, TIMESTAMP '2024-01-01 00:00:00'),
                (1, TIMESTAMP '2024-01-02 00:00:00'),
                (2, TIMESTAMP '2024-01-01 00:00:00')
            AS t(merchant_id, created_at)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(path)],
    )

    assert first_sorting_violation(path) is None


def test_checker_reports_first_sorting_violation(tmp_path):
    path = tmp_path / "unsorted.parquet"
    write_transactions_fixture(path)

    violation = first_sorting_violation(path)

    assert violation is not None
    assert violation[0] == 2
    assert violation[1] == 2
    assert violation[2] == 1

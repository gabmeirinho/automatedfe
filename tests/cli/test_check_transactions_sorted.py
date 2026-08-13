import duckdb

from automatedfe.cli.check_transactions_sorted import main
from tests.conftest import write_transactions_fixture


def test_checker_accepts_sorted_transactions(tmp_path, capsys):
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

    assert main(["--input", str(path)]) == 0
    assert capsys.readouterr().out == (
        f"PASS: {path} is sorted by merchant_id, created_at\n"
    )


def test_checker_reports_first_sorting_violation(tmp_path, capsys):
    path = tmp_path / "unsorted.parquet"
    write_transactions_fixture(path)

    assert main(["--input", str(path)]) == 1
    assert capsys.readouterr().out.splitlines() == [
        f"FAIL: {path} is not sorted",
        "First violation at row 2:",
        "  previous: merchant_id=2, created_at=2024-01-02 00:00:00",
        "  current:  merchant_id=1, created_at=2024-01-03 00:00:00",
    ]

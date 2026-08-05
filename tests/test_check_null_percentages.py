import subprocess
import sys

import duckdb

from scripts.check_null_percentages import DEFAULT_INPUT_DIR


SCRIPT = "scripts/check_null_percentages.py"


def test_checker_uses_transformed_datasets_by_default(tmp_path):
    transformed_dir = tmp_path / "transformed"
    transformed_dir.mkdir()

    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (1, 'visa', NULL),
                (2, NULL, 'cpf')
            AS t(merchant_id, card_brand, document_type)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(transformed_dir / "transactions.parquet")],
    )

    result = subprocess.run(
        [sys.executable, SCRIPT, "--input-dir", str(transformed_dir)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "dataset\tcolumn\trows\tnull_count\tnull_percent",
        "transactions.parquet\tmerchant_id\t2\t0\t0.000000%",
        "transactions.parquet\tcard_brand\t2\t1\t50.000000%",
        "transactions.parquet\tdocument_type\t2\t1\t50.000000%",
    ]
    assert DEFAULT_INPUT_DIR.name == "transformed"


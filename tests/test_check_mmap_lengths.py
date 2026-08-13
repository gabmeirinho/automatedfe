import subprocess
import sys

from automatedfe.data.transaction_materialization import materialize_transactions
from tests.test_materialization import write_transformed_fixture

SCRIPT = "scripts/check_mmap_lengths.py"


def run_checker(transformed_path, mmap_dir):
    return subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--transformed",
            str(transformed_path),
            "--mmap-dir",
            str(mmap_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_checker_accepts_mmaps_with_matching_lengths(tmp_path):
    transformed_path = write_transformed_fixture(tmp_path)
    mmap_dir = tmp_path / "mmap"
    materialize_transactions(transformed_path, mmap_dir)

    result = run_checker(transformed_path, mmap_dir)

    assert result.returncode == 0
    assert "PASS: all 5 mmap files contain 5 rows" in result.stdout


def test_checker_rejects_a_mmap_with_a_different_length(tmp_path):
    transformed_path = write_transformed_fixture(tmp_path)
    mmap_dir = tmp_path / "mmap"
    materialize_transactions(transformed_path, mmap_dir)
    mmap_path = mmap_dir / "amount.mmap"
    mmap_path.write_bytes(mmap_path.read_bytes()[:-8])

    result = run_checker(transformed_path, mmap_dir)

    assert result.returncode == 1
    assert "amount.mmap: 4 rows (MISMATCH)" in result.stdout
    assert "FAIL: mmap lengths do not match" in result.stdout

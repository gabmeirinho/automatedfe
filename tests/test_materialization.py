
import duckdb
import numpy as np
import pytest

from automatedfe.features import (
    Aggregation,
    FeatureSpec,
    RowWindow,
    materialize_feature,
)
from automatedfe.transaction_materialization import (
    MANIFEST_FILENAME,
    MMAP_SUFFIX,
    column_dtype,
    load_mmapped_columns,
    materialize_transactions,
    read_manifest,
)


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


def test_column_dtype_mapping():
    assert column_dtype("BIGINT").name == "int64"
    assert column_dtype("INTEGER").name == "int32"
    assert column_dtype("DOUBLE").name == "float64"
    assert column_dtype("FLOAT").name == "float32"
    assert column_dtype("TIMESTAMP WITH TIME ZONE").name == "int64"
    assert column_dtype("TIMESTAMP").name == "int64"
    assert column_dtype("DATE").name == "int32"
    assert column_dtype("DECIMAL(18,3)").name == "float64"
    assert column_dtype("VARCHAR") is None
    assert column_dtype("BOOLEAN") is None


def test_materialize_writes_only_numeric_columns(tmp_path):
    input_path = write_transformed_fixture(tmp_path)
    output_dir = tmp_path / "mmap"

    materialize_transactions(input_path, output_dir, chunk_size=2)

    materialized = {path.stem for path in output_dir.iterdir() if path.suffix == MMAP_SUFFIX}
    assert materialized == {"merchant_id", "amount", "status", "created_at", "card_token_id"}
    assert not (output_dir / f"merchant_category_code{MMAP_SUFFIX}").exists()


def test_materialize_writes_correct_values(tmp_path):
    input_path = write_transformed_fixture(tmp_path)
    output_dir = tmp_path / "mmap"

    materialize_transactions(input_path, output_dir, chunk_size=2)

    merchant_ids = np.memmap(output_dir / "merchant_id.mmap", dtype="int64", mode="r", shape=(5,))
    amounts = np.memmap(output_dir / "amount.mmap", dtype="float64", mode="r", shape=(5,))
    statuses = np.memmap(output_dir / "status.mmap", dtype="int32", mode="r", shape=(5,))
    created_at = np.memmap(output_dir / "created_at.mmap", dtype="int64", mode="r", shape=(5,))

    np.testing.assert_array_equal(merchant_ids, [1, 1, 2, 2, 3])
    np.testing.assert_allclose(amounts, [100.5, 200.0, 50.25, 75.0, 10.0])
    np.testing.assert_array_equal(statuses, [0, 1, 0, 2, 1])
    np.testing.assert_array_equal(
        created_at,
        [1704067200000000, 1704153600000000, 1706745600000000, 1707091200000000, 1709251200000000],
    )


def test_materialize_writes_manifest(tmp_path):
    input_path = write_transformed_fixture(tmp_path)
    output_dir = tmp_path / "mmap"

    materialize_transactions(input_path, output_dir)

    manifest = read_manifest(output_dir)
    assert manifest["rows"] == 5
    assert set(manifest["columns"]) == {
        "merchant_id",
        "amount",
        "status",
        "created_at",
        "card_token_id",
    }
    assert manifest["columns"]["merchant_id"] == {
        "file": "merchant_id.mmap",
        "dtype": "int64",
    }
    assert manifest["columns"]["status"] == {"file": "status.mmap", "dtype": "int32"}
    assert (output_dir / MANIFEST_FILENAME).exists()


def test_load_mmapped_columns_reads_back(tmp_path):
    input_path = write_transformed_fixture(tmp_path)
    output_dir = tmp_path / "mmap"

    materialize_transactions(input_path, output_dir)

    columns = load_mmapped_columns(output_dir)
    assert list(columns) == ["merchant_id", "amount", "status", "created_at", "card_token_id"]
    assert columns["merchant_id"].dtype == np.dtype("int64")
    np.testing.assert_array_equal(columns["merchant_id"], [1, 1, 2, 2, 3])

    subset = load_mmapped_columns(output_dir, columns=["status", "amount"])
    assert list(subset) == ["status", "amount"]
    np.testing.assert_array_equal(subset["status"], [0, 1, 0, 2, 1])


def test_load_mmapped_columns_rejects_unknown_column(tmp_path):
    input_path = write_transformed_fixture(tmp_path)
    output_dir = tmp_path / "mmap"
    materialize_transactions(input_path, output_dir)

    with pytest.raises(ValueError, match="merchant_category_code"):
        load_mmapped_columns(output_dir, columns=["merchant_category_code"])


def test_materialize_requires_force_for_existing_output(tmp_path):
    input_path = write_transformed_fixture(tmp_path)
    output_dir = tmp_path / "mmap"
    materialize_transactions(input_path, output_dir)

    with pytest.raises(FileExistsError, match="already exists"):
        materialize_transactions(input_path, output_dir)


def test_materialize_force_replaces_existing_output(tmp_path):
    input_path = write_transformed_fixture(tmp_path)
    output_dir = tmp_path / "mmap"
    materialize_transactions(input_path, output_dir)
    stale = output_dir / f"stale{MMAP_SUFFIX}"
    stale.touch()

    materialize_transactions(input_path, output_dir, force=True)

    assert not stale.exists()
    assert (output_dir / "amount.mmap").exists()


def test_materialize_chunk_size_is_validated(tmp_path):
    input_path = write_transformed_fixture(tmp_path)
    output_dir = tmp_path / "mmap"

    with pytest.raises(ValueError, match="chunk_size"):
        materialize_transactions(input_path, output_dir, chunk_size=0)


def test_materialize_rejects_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        materialize_transactions(tmp_path / "missing.parquet", tmp_path / "mmap")


def test_materialize_rejects_parquet_without_numeric_columns(tmp_path):
    path = tmp_path / "strings.parquet"
    duckdb.sql(
        "COPY (SELECT 'a' AS name, 'b' AS code) TO ? (FORMAT PARQUET)",
        params=[str(path)],
    )

    with pytest.raises(ValueError, match="no materializable"):
        materialize_transactions(path, tmp_path / "mmap")


def test_materialize_feature_uses_transaction_mmaps_and_can_write_derived_mmap(tmp_path):
    columns = {
        "merchant_id": np.array([1, 1, 1, 2], dtype=np.int64),
        "amount": np.array([10.0, 20.0, 30.0, 100.0]),
        "created_at": np.array([1, 2, 3, 1], dtype=np.int64),
    }
    spec = FeatureSpec(Aggregation.MEAN, "amount", RowWindow(2))
    output_path = tmp_path / "mean_amount_last_2_rows.mmap"

    result = materialize_feature(spec, columns, output_path=output_path)

    assert isinstance(result, np.memmap)
    np.testing.assert_allclose(result, [np.nan, 10.0, 15.0, np.nan], equal_nan=True)
    reopened = np.memmap(output_path, dtype=np.float64, mode="r", shape=(4,))
    np.testing.assert_allclose(reopened, result, equal_nan=True)


def test_materialize_feature_rejects_missing_required_mmap_column():
    spec = FeatureSpec(Aggregation.MEAN, "amount", RowWindow(2))

    with pytest.raises(ValueError, match="missing column.*amount"):
        materialize_feature(spec, {"merchant_id": np.array([1, 1])})

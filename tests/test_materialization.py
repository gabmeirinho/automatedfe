
import json

import duckdb
import numpy as np
import pytest

import automatedfe.features.feature_materialization as feature_materialization
from automatedfe.features import (
    FeatureMaterializer,
    RowWindow,
    TxFeature,
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
    spec = TxFeature("mean", "amount", RowWindow(2).rows, "row")
    output_path = tmp_path / "mean_amount_last_2_rows.mmap"

    result = materialize_feature(spec, columns, output_path=output_path)

    assert isinstance(result, np.memmap)
    np.testing.assert_allclose(result, [np.nan, 10.0, 15.0, np.nan], equal_nan=True)
    reopened = np.memmap(output_path, dtype=np.float64, mode="r", shape=(4,))
    np.testing.assert_allclose(reopened, result, equal_nan=True)


def test_materialize_feature_rejects_missing_required_mmap_column():
    spec = TxFeature("mean", "amount", RowWindow(2).rows, "row")

    with pytest.raises(ValueError, match="missing column.*amount"):
        materialize_feature(spec, {"merchant_id": np.array([1, 1])})


def test_feature_materializer_reuses_a_feature_within_a_run(monkeypatch, tmp_path):
    columns = {
        "merchant_id": np.array([1, 1, 1, 2], dtype=np.int64),
        "amount": np.array([10.0, 20.0, 30.0, 100.0]),
        "created_at": np.array([1, 2, 3, 1], dtype=np.int64),
    }
    spec = TxFeature("mean", "amount", RowWindow(2).rows, "row")
    calls = 0
    original = feature_materialization._compute_primitive

    def counted_materialize(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(feature_materialization, "_compute_primitive", counted_materialize)
    materializer = FeatureMaterializer(columns, output_dir=tmp_path / "features")

    first = materializer.materialize(spec)
    second = materializer.materialize(spec)

    assert calls == 1
    assert first is second
    np.testing.assert_allclose(first, [np.nan, 10.0, 15.0, np.nan], equal_nan=True)


def test_feature_materializer_deduplicates_a_population(monkeypatch):
    columns = {
        "merchant_id": np.array([1, 1], dtype=np.int64),
        "amount": np.array([10.0, 20.0]),
    }
    spec = TxFeature("mean", "amount", RowWindow(2).rows, "row")
    calls = 0
    original = feature_materialization._compute_primitive

    def counted_materialize(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(feature_materialization, "_compute_primitive", counted_materialize)
    FeatureMaterializer(columns).materialize_population([spec, spec])

    assert calls == 1


def test_event_features_are_cached_on_disk_and_reused_across_runs(tmp_path):
    columns = {
        "merchant_id": np.array([1, 1, 1, 2, 2], dtype=np.int64),
        "created_at": np.array([1, 2, 3, 1, 5], dtype=np.int64),
        "amount": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
    }
    events_merchants = np.array([1, 2], dtype=np.int64)
    events_timestamps = np.array([4, 6], dtype=np.int64)
    spec = TxFeature("sum", "amount", RowWindow(2).rows, "row")
    features_dir = tmp_path / "features"

    first = FeatureMaterializer(columns, features_dir=features_dir)
    values = first.materialize_for_events(spec, events_merchants, events_timestamps)
    np.testing.assert_allclose(values, [50.0, 90.0])

    files = {path.name for path in features_dir.iterdir()}
    assert f"{spec.name}.events.mmap" in files
    assert f"{spec.name}.events.json" in files

    # A brand-new materializer over the same event set must load from disk
    # instead of recomputing.
    second = FeatureMaterializer(columns, features_dir=features_dir)
    cached = second.materialize_for_events(spec, events_merchants, events_timestamps)
    np.testing.assert_allclose(cached, [50.0, 90.0])


def test_event_feature_disk_cache_invalidates_on_event_set_change(tmp_path):
    columns = {
        "merchant_id": np.array([1, 1, 1, 2, 2], dtype=np.int64),
        "created_at": np.array([1, 2, 3, 1, 5], dtype=np.int64),
        "amount": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
    }
    spec = TxFeature("sum", "amount", RowWindow(2).rows, "row")
    features_dir = tmp_path / "features"

    first = FeatureMaterializer(columns, features_dir=features_dir)
    first_values = first.materialize_for_events(
        spec,
        np.array([1, 2], dtype=np.int64),
        np.array([2, 3], dtype=np.int64),
    )
    np.testing.assert_allclose(first_values, [10.0, 40.0])

    # A different event set (same size, different timestamps) must recompute.
    second = FeatureMaterializer(columns, features_dir=features_dir)
    second_values = second.materialize_for_events(
        spec,
        np.array([1, 2], dtype=np.int64),
        np.array([10, 11], dtype=np.int64),
    )
    np.testing.assert_allclose(second_values, [50.0, 90.0])

    # And the original event set is still correct after the overwrite.
    third = FeatureMaterializer(columns, features_dir=features_dir)
    third_values = third.materialize_for_events(
        spec,
        np.array([1, 2], dtype=np.int64),
        np.array([2, 3], dtype=np.int64),
    )
    np.testing.assert_allclose(third_values, [10.0, 40.0])


def test_event_features_are_not_persisted_without_features_dir(tmp_path):
    columns = {
        "merchant_id": np.array([1, 1], dtype=np.int64),
        "created_at": np.array([1, 2], dtype=np.int64),
        "amount": np.array([10.0, 20.0]),
    }
    spec = TxFeature("count_total", None, RowWindow(2).rows, "row")

    FeatureMaterializer(columns).materialize_for_events(
        spec,
        np.array([1], dtype=np.int64),
        np.array([3], dtype=np.int64),
    )

    assert list(tmp_path.iterdir()) == []


def test_materialize_with_duration_reuses_identical_duration(monkeypatch):
    columns = {
        "merchant_id": np.array([1, 1, 1, 2], dtype=np.int64),
        "amount": np.array([10.0, 20.0, 30.0, 100.0]),
        "created_at": np.array([1, 2, 3, 1], dtype=np.int64),
    }
    spec = TxFeature("mean", "amount", RowWindow(2).rows, "row")
    calls = 0
    original = feature_materialization._compute_primitive

    def counted_materialize(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(feature_materialization, "_compute_primitive", counted_materialize)
    materializer = FeatureMaterializer(columns)

    first_values, first_duration = materializer.materialize_with_duration(spec)
    second_values, second_duration = materializer.materialize_with_duration(spec)

    assert calls == 1
    assert first_values is second_values
    assert first_duration >= 0.0
    assert first_duration == second_duration
    np.testing.assert_allclose(first_values, [np.nan, 10.0, 15.0, np.nan], equal_nan=True)


def test_materialize_with_duration_preserves_ndarray_api(tmp_path):
    columns = {
        "merchant_id": np.array([1, 1, 1, 2], dtype=np.int64),
        "amount": np.array([10.0, 20.0, 30.0, 100.0]),
        "created_at": np.array([1, 2, 3, 1], dtype=np.int64),
    }
    spec = TxFeature("mean", "amount", RowWindow(2).rows, "row")
    materializer = FeatureMaterializer(columns, output_dir=tmp_path / "features")

    values, duration = materializer.materialize_with_duration(spec)

    assert isinstance(values, np.memmap)
    assert duration >= 0.0
    assert materializer.materialize(spec) is values


def test_event_feature_duration_persisted_and_reused_from_disk(tmp_path):
    columns = {
        "merchant_id": np.array([1, 1, 1, 2, 2], dtype=np.int64),
        "created_at": np.array([1, 2, 3, 1, 5], dtype=np.int64),
        "amount": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
    }
    events_merchants = np.array([1, 2], dtype=np.int64)
    events_timestamps = np.array([4, 6], dtype=np.int64)
    spec = TxFeature("sum", "amount", RowWindow(2).rows, "row")
    features_dir = tmp_path / "features"

    first = FeatureMaterializer(columns, features_dir=features_dir)
    values, first_duration = first.materialize_for_events_with_duration(
        spec, events_merchants, events_timestamps
    )
    np.testing.assert_allclose(values, [50.0, 90.0])
    assert first_duration >= 0.0

    metadata = json.loads(
        (features_dir / f"{spec.name}.events.json").read_text()
    )
    assert metadata["duration"] == first_duration

    second = FeatureMaterializer(columns, features_dir=features_dir)
    cached_values, cached_duration = second.materialize_for_events_with_duration(
        spec, events_merchants, events_timestamps
    )
    np.testing.assert_allclose(cached_values, [50.0, 90.0])
    assert cached_duration == first_duration

    again_values, again_duration = second.materialize_for_events_with_duration(
        spec, events_merchants, events_timestamps
    )
    np.testing.assert_allclose(again_values, [50.0, 90.0])
    assert again_duration == cached_duration


def test_event_feature_legacy_metadata_without_duration_is_recomputed_once(
    tmp_path, monkeypatch
):
    columns = {
        "merchant_id": np.array([1, 1, 1, 2, 2], dtype=np.int64),
        "created_at": np.array([1, 2, 3, 1, 5], dtype=np.int64),
        "amount": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
    }
    events_merchants = np.array([1, 2], dtype=np.int64)
    events_timestamps = np.array([4, 6], dtype=np.int64)
    spec = TxFeature("sum", "amount", RowWindow(2).rows, "row")
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    checksum = FeatureMaterializer._event_set_checksum(
        events_merchants, events_timestamps
    )
    legacy_values = np.array([50.0, 90.0])
    mapped = np.memmap(
        features_dir / f"{spec.name}.events.mmap",
        dtype=np.float64,
        mode="w+",
        shape=legacy_values.shape,
    )
    mapped[:] = legacy_values
    mapped.flush()
    (features_dir / f"{spec.name}.events.json").write_text(
        json.dumps({"name": spec.name, "rows": 2, "checksum": checksum})
    )

    calls = 0
    original = feature_materialization._compute_primitive

    def counted_materialize(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(feature_materialization, "_compute_primitive", counted_materialize)
    materializer = FeatureMaterializer(columns, features_dir=features_dir)

    values, duration = materializer.materialize_for_events_with_duration(
        spec, events_merchants, events_timestamps
    )
    np.testing.assert_allclose(values, [50.0, 90.0])
    assert calls == 1
    assert duration >= 0.0

    repaired = json.loads((features_dir / f"{spec.name}.events.json").read_text())
    assert repaired["duration"] == duration

    calls = 0
    second = FeatureMaterializer(columns, features_dir=features_dir)
    cached_values, cached_duration = second.materialize_for_events_with_duration(
        spec, events_merchants, events_timestamps
    )
    assert calls == 0
    assert cached_duration == duration
    np.testing.assert_allclose(cached_values, [50.0, 90.0])


def test_failed_materialization_does_not_record_timing(tmp_path):
    columns = {
        "merchant_id": np.array([1, 1], dtype=np.int64),
        "created_at": np.array([1, 2], dtype=np.int64),
    }
    spec = TxFeature("mean", "amount", RowWindow(2).rows, "row")
    features_dir = tmp_path / "features"
    materializer = FeatureMaterializer(columns, features_dir=features_dir)

    with pytest.raises(ValueError, match="missing column.*amount"):
        materializer.materialize_for_events_with_duration(
            spec,
            np.array([1], dtype=np.int64),
            np.array([3], dtype=np.int64),
        )
    assert not features_dir.exists()

    count_spec = TxFeature("count_total", None, RowWindow(2).rows, "row")
    count_values, count_duration = materializer.materialize_for_events_with_duration(
        count_spec,
        np.array([1], dtype=np.int64),
        np.array([3], dtype=np.int64),
    )
    assert count_duration >= 0.0
    np.testing.assert_allclose(count_values, [2.0])

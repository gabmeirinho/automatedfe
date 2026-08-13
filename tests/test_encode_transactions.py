from pathlib import Path

import duckdb
import pytest

from automatedfe.data.encoding import (
    encode_transactions,
    fit_label_mapping,
    load_label_mapping,
)


def write_fixtures(tmp_path):
    transactions_path = tmp_path / "transactions.parquet"
    dataset_path = tmp_path / "dataset.parquet"

    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (10, 'approved', 'visa', '5812', 'cnpj', 'online', 'pending'),
                (10, 'declined', 'visa', '5812', 'cnpj', 'online', 'settled'),
                (10, 'approved', 'mastercard', '5812', 'cnpj', 'chip', 'pending'),
                (20, 'approved', 'visa', '5812', 'cnpj', 'online', 'pending'),
                (30, 'refunded', '0', '5812', 'cnpj', 'online', 'pending')
            AS t(
                merchant_id,
                status,
                card_brand,
                merchant_category_code,
                document_type,
                capture_method,
                payment_method
            )
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(transactions_path)],
    )
    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (10, 'train'),
                (20, 'test')
            AS t(merchant_id, split)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(dataset_path)],
    )
    return transactions_path, dataset_path


def read_rows(path: Path, columns: str = "merchant_id, status, card_brand"):
    return duckdb.sql(
        f"SELECT {columns} FROM read_parquet(?)",
        params=[str(path)],
    ).fetchall()


def test_fit_label_mapping_uses_train_split_only(tmp_path):
    transactions_path, dataset_path = write_fixtures(tmp_path)
    mapping_path = tmp_path / "label_mapping.json"

    mapping = fit_label_mapping(
        transactions_path,
        dataset_path,
        mapping_path,
        columns=["status", "card_brand"],
    )

    # Only merchants in the train split contribute values: merchant 20 (test)
    # and merchant 30 (absent from the dataset) are excluded.
    assert mapping == {
        "status": {"approved": 0, "declined": 1},
        "card_brand": {"mastercard": 0, "visa": 1},
    }
    assert load_label_mapping(mapping_path) == mapping


def test_fit_label_mapping_default_columns(tmp_path):
    transactions_path, dataset_path = write_fixtures(tmp_path)
    mapping_path = tmp_path / "label_mapping.json"

    mapping = fit_label_mapping(transactions_path, dataset_path, mapping_path)

    assert set(mapping) == {
        "status",
        "card_brand",
        "document_type",
        "capture_method",
        "payment_method",
    }


def test_fit_label_mapping_requires_force_for_existing_output(tmp_path):
    transactions_path, dataset_path = write_fixtures(tmp_path)
    mapping_path = tmp_path / "label_mapping.json"
    mapping_path.touch()

    with pytest.raises(FileExistsError, match="already exists"):
        fit_label_mapping(transactions_path, dataset_path, mapping_path)


def test_fit_label_mapping_rejects_missing_input(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    duckdb.sql(
        "COPY (SELECT 1 AS merchant_id, 'train' AS split) TO ? (FORMAT PARQUET)",
        params=[str(dataset_path)],
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        fit_label_mapping(tmp_path / "missing.parquet", dataset_path, "x.json")


def test_fit_label_mapping_rejects_missing_column(tmp_path):
    transactions_path, dataset_path = write_fixtures(tmp_path)

    with pytest.raises(ValueError, match="missing_column"):
        fit_label_mapping(
            transactions_path,
            dataset_path,
            tmp_path / "mapping.json",
            columns=["status", "missing_column"],
        )


def test_fit_label_mapping_rejects_missing_dataset_split(tmp_path):
    transactions_path, _ = write_fixtures(tmp_path)
    dataset_path = tmp_path / "dataset.parquet"
    duckdb.sql(
        "COPY (SELECT 1 AS merchant_id) TO ? (FORMAT PARQUET)",
        params=[str(dataset_path)],
    )

    with pytest.raises(ValueError, match="split"):
        fit_label_mapping(
            transactions_path,
            dataset_path,
            tmp_path / "mapping.json",
            columns=["status"],
        )


def test_encode_transactions_applies_mapping_and_reserves_unknown(tmp_path):
    transactions_path, dataset_path = write_fixtures(tmp_path)
    mapping_path = tmp_path / "label_mapping.json"
    output_path = tmp_path / "encoded.parquet"

    mapping = fit_label_mapping(
        transactions_path,
        dataset_path,
        mapping_path,
        columns=["status", "card_brand"],
    )
    encode_transactions(transactions_path, output_path, mapping)

    assert read_rows(output_path) == [
        (10, 0, 1),
        (10, 1, 1),
        (10, 0, 0),
        (20, 0, 1),
        (30, 2, 2),
    ]


def test_encode_transactions_accepts_mapping_path(tmp_path):
    transactions_path, dataset_path = write_fixtures(tmp_path)
    mapping_path = tmp_path / "label_mapping.json"
    output_path = tmp_path / "encoded.parquet"

    fit_label_mapping(
        transactions_path,
        dataset_path,
        mapping_path,
        columns=["status", "card_brand"],
    )
    encode_transactions(transactions_path, output_path, mapping_path)

    assert len(read_rows(output_path)) == 5


def test_encode_transactions_preserves_other_columns(tmp_path):
    transactions_path, dataset_path = write_fixtures(tmp_path)
    mapping_path = tmp_path / "label_mapping.json"
    output_path = tmp_path / "encoded.parquet"

    fit_label_mapping(
        transactions_path,
        dataset_path,
        mapping_path,
        columns=["status", "card_brand"],
    )
    encode_transactions(transactions_path, output_path, mapping_path)

    assert [row[0] for row in read_rows(output_path, "merchant_id")] == [10, 10, 10, 20, 30]


def test_encode_transactions_requires_force_for_existing_output(tmp_path):
    transactions_path, dataset_path = write_fixtures(tmp_path)
    mapping_path = tmp_path / "label_mapping.json"
    output_path = tmp_path / "encoded.parquet"
    fit_label_mapping(
        transactions_path,
        dataset_path,
        mapping_path,
        columns=["status"],
    )
    output_path.touch()

    with pytest.raises(FileExistsError, match="already exists"):
        encode_transactions(transactions_path, output_path, mapping_path)


def test_encode_transactions_applies_in_place(tmp_path):
    transactions_path, dataset_path = write_fixtures(tmp_path)
    mapping_path = tmp_path / "label_mapping.json"
    fit_label_mapping(
        transactions_path,
        dataset_path,
        mapping_path,
        columns=["status", "card_brand"],
    )

    encode_transactions(transactions_path, transactions_path, mapping_path, force=True)

    assert read_rows(transactions_path) == [
        (10, 0, 1),
        (10, 1, 1),
        (10, 0, 0),
        (20, 0, 1),
        (30, 2, 2),
    ]


def test_encode_transactions_in_place_requires_force(tmp_path):
    transactions_path, dataset_path = write_fixtures(tmp_path)
    mapping_path = tmp_path / "label_mapping.json"
    fit_label_mapping(
        transactions_path,
        dataset_path,
        mapping_path,
        columns=["status", "card_brand"],
    )

    with pytest.raises(FileExistsError, match="already exists"):
        encode_transactions(transactions_path, transactions_path, mapping_path)


def test_encode_transactions_preserves_merchant_category_code(tmp_path):
    transactions_path, dataset_path = write_fixtures(tmp_path)
    mapping_path = tmp_path / "label_mapping.json"
    output_path = tmp_path / "encoded.parquet"

    mapping = fit_label_mapping(
        transactions_path,
        dataset_path,
        mapping_path,
        columns=["status", "card_brand"],
    )
    encode_transactions(transactions_path, output_path, mapping)

    assert [row[0] for row in read_rows(output_path, "merchant_category_code")] == [
        "5812",
        "5812",
        "5812",
        "5812",
        "5812",
    ]


def test_encode_transactions_rejects_missing_mapping_column(tmp_path):
    transactions_path, dataset_path = write_fixtures(tmp_path)
    mapping_path = tmp_path / "label_mapping.json"
    fit_label_mapping(
        transactions_path,
        dataset_path,
        mapping_path,
        columns=["status"],
    )
    missing_path = tmp_path / "missing.parquet"
    duckdb.sql(
        "COPY (SELECT 1 AS merchant_id) TO ? (FORMAT PARQUET)",
        params=[str(missing_path)],
    )

    with pytest.raises(ValueError, match="status"):
        encode_transactions(missing_path, tmp_path / "encoded.parquet", mapping_path)

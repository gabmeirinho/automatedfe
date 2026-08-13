from pathlib import Path

import duckdb

from automatedfe.data.preprocessing import preprocess


def write_pipeline_fixtures(tmp_path):
    transactions_path = tmp_path / "transactions.parquet"
    card_tokens_path = tmp_path / "card_tokens.parquet"
    merchants_path = tmp_path / "merchants.parquet"
    dataset_path = tmp_path / "dataset.parquet"

    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (1, TIMESTAMP '2024-01-02 00:00:00', 100, 'approved', 'online', 'pending'),
                (2, TIMESTAMP '2024-01-01 00:00:00', 300, 'declined', 'chip', 'settled'),
                (1, TIMESTAMP '2024-01-01 00:00:00', 200, 'approved', 'chip', 'settled'),
                (3, TIMESTAMP '2024-01-05 00:00:00', 400, 'refunded', 'app', 'pending')
            AS t(merchant_id, created_at, card_token_id, status, capture_method, payment_method)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(transactions_path)],
    )
    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (100, 'visa'),
                (200, 'mastercard'),
                (300, 'visa'),
                (400, 'amex')
            AS t(id, card_brand)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(card_tokens_path)],
    )
    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (1, 'cnpj'),
                (2, 'cpf')
            AS t(id, document_type)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(merchants_path)],
    )
    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (2, TIMESTAMP '2024-01-02 00:00:00', 'test'),
                (1, TIMESTAMP '2024-01-01 00:00:00', 'train')
            AS t(merchant_id, event_timestamp, split)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(dataset_path)],
    )
    return {
        "transactions": transactions_path,
        "card_tokens": card_tokens_path,
        "merchants": merchants_path,
        "dataset": dataset_path,
    }


def read_rows(path: Path, columns: str = "*"):
    return duckdb.sql(
        f"SELECT {columns} FROM read_parquet(?)",
        params=[str(path)],
    ).fetchall()


def test_preprocess_runs_end_to_end(tmp_path):
    paths = write_pipeline_fixtures(tmp_path)
    transformed_path = tmp_path / "transformed" / "transactions.parquet"
    dataset_output_path = tmp_path / "transformed" / "dataset.parquet"
    mapping_path = tmp_path / "transformed" / "label_mapping.json"
    mmap_dir = tmp_path / "transformed" / "mmap"

    preprocess(
        paths["transactions"],
        transformed_path,
        card_tokens_path=paths["card_tokens"],
        merchants_path=paths["merchants"],
        dataset_path=paths["dataset"],
        dataset_output_path=dataset_output_path,
        mapping_path=mapping_path,
        mmap_dir=mmap_dir,
    )

    assert read_rows(
        dataset_output_path, "merchant_id, CAST(event_timestamp AS VARCHAR), split"
    ) == [
        (1, "2024-01-01 00:00:00", "train"),
        (2, "2024-01-02 00:00:00", "test"),
    ]

    assert read_rows(
        transformed_path,
        "merchant_id, status, capture_method, payment_method, card_brand, document_type",
    ) == [
        (1, 0, 0, 1, 0, 0),
        (1, 0, 1, 0, 1, 0),
        (2, 1, 0, 1, 1, 1),
        (3, 1, 2, 0, 2, 1),
    ]

    assert mapping_path.exists()
    assert (mmap_dir / "manifest.json").exists()
    assert (mmap_dir / "merchant_id.mmap").exists()
    assert not (mmap_dir / "merchant_category_code.mmap").exists()


def test_preprocess_requires_force_for_existing_sort_outputs(tmp_path):
    paths = write_pipeline_fixtures(tmp_path)
    dataset_output_path = tmp_path / "transformed" / "dataset.parquet"
    dataset_output_path.parent.mkdir(parents=True)
    dataset_output_path.touch()

    try:
        preprocess(
            paths["transactions"],
            tmp_path / "transformed" / "transactions.parquet",
            card_tokens_path=paths["card_tokens"],
            merchants_path=paths["merchants"],
            dataset_path=paths["dataset"],
            dataset_output_path=dataset_output_path,
            mapping_path=tmp_path / "transformed" / "label_mapping.json",
        )
    except FileExistsError:
        assert True
    else:
        raise AssertionError("expected preprocess to reject existing sort outputs")


def test_preprocess_force_replaces_existing_outputs(tmp_path):
    paths = write_pipeline_fixtures(tmp_path)
    transformed_path = tmp_path / "transformed" / "transactions.parquet"
    dataset_output_path = tmp_path / "transformed" / "dataset.parquet"
    mapping_path = tmp_path / "transformed" / "label_mapping.json"
    mmap_dir = tmp_path / "transformed" / "mmap"

    preprocess(
        paths["transactions"],
        transformed_path,
        card_tokens_path=paths["card_tokens"],
        merchants_path=paths["merchants"],
        dataset_path=paths["dataset"],
        dataset_output_path=dataset_output_path,
        mapping_path=mapping_path,
        mmap_dir=mmap_dir,
    )
    preprocess(
        paths["transactions"],
        transformed_path,
        card_tokens_path=paths["card_tokens"],
        merchants_path=paths["merchants"],
        dataset_path=paths["dataset"],
        dataset_output_path=dataset_output_path,
        mapping_path=mapping_path,
        mmap_dir=mmap_dir,
        force=True,
    )

    assert len(read_rows(transformed_path)) == 4

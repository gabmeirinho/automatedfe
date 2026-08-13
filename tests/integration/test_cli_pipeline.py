import json

import duckdb

from automatedfe.cli.main import main as dispatch


def write_pipeline_fixtures(tmp_path):
    transactions_path = tmp_path / "transactions.parquet"
    card_tokens_path = tmp_path / "card_tokens.parquet"
    merchants_path = tmp_path / "merchants.parquet"
    dataset_path = tmp_path / "dataset.parquet"

    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (1, TIMESTAMPTZ '2024-01-01 00:00:00+00:00', 101, 10.0, 'approved', 'online', 'pending'),
                (2, TIMESTAMPTZ '2024-01-01 00:00:00+00:00', 102, 20.0, 'declined', 'chip', 'settled'),
                (3, TIMESTAMPTZ '2024-01-01 00:00:00+00:00', 103, 30.0, 'approved', 'online', 'pending'),
                (4, TIMESTAMPTZ '2024-01-01 00:00:00+00:00', 104, 40.0, 'declined', 'chip', 'settled')
            AS t(
                merchant_id,
                created_at,
                card_token_id,
                amount,
                status,
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
                (101, 'visa'),
                (102, 'mastercard'),
                (103, 'visa'),
                (104, 'mastercard')
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
                (2, 'cpf'),
                (3, 'cnpj'),
                (4, 'cpf')
            AS t(id, document_type)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(merchants_path)],
    )
    duckdb.sql(
        """
        COPY (
            SELECT * FROM VALUES
                (1, TIMESTAMPTZ '2024-01-02 00:00:00+00:00', 0, 'train'),
                (2, TIMESTAMPTZ '2024-01-02 00:00:00+00:00', 1, 'train'),
                (3, TIMESTAMPTZ '2024-01-02 00:00:00+00:00', 0, 'test'),
                (4, TIMESTAMPTZ '2024-01-02 00:00:00+00:00', 1, 'test')
            AS t(merchant_id, event_timestamp, label, split)
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


def test_dispatch_runs_preprocessing_and_bounded_search(tmp_path):
    paths = write_pipeline_fixtures(tmp_path)
    transformed_dir = tmp_path / "transformed"
    transformed_path = transformed_dir / "transactions.parquet"
    dataset_output_path = transformed_dir / "dataset.parquet"
    mapping_path = transformed_dir / "label_mapping.json"
    mmap_dir = transformed_dir / "mmap"
    summary_path = tmp_path / "search-summary.json"

    assert dispatch(
        [
            "preprocess",
            "--transactions",
            str(paths["transactions"]),
            "--transformed",
            str(transformed_path),
            "--card-tokens",
            str(paths["card_tokens"]),
            "--merchants",
            str(paths["merchants"]),
            "--dataset",
            str(paths["dataset"]),
            "--dataset-output",
            str(dataset_output_path),
            "--mapping",
            str(mapping_path),
            "--mmap-dir",
            str(mmap_dir),
        ]
    ) == 0

    assert dispatch(
        [
            "search",
            "--strategy",
            "enumerative_without_archive",
            "--candidate-count",
            "1",
            "--max-depth",
            "1",
            "--dataset",
            str(dataset_output_path),
            "--mapping",
            str(mapping_path),
            "--mmap-dir",
            str(mmap_dir),
            "--summary",
            str(summary_path),
        ]
    ) == 0

    assert transformed_path.is_file()
    assert (mmap_dir / "manifest.json").is_file()
    summary = json.loads(summary_path.read_text())
    assert summary["strategy"] == "enumerative_without_archive"
    assert summary["counts"]["generated"] == 1
    assert summary["counts"]["evaluated"] == 0
    assert summary["selected_feature_count"] == 1

from datetime import datetime, timedelta, timezone

import duckdb
import numpy as np
import pytest

from automatedfe.features import (
    Aggregation,
    FeatureMaterializer,
    FeatureSpec,
    RowWindow,
)
from automatedfe.features.fitness import LogisticRegressionFitness


def write_temporal_dataset(path):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(12):
        rows.append((1, base + timedelta(days=index + 1), index % 2, "train"))
    # These rows deliberately sit outside the training split. They must never
    # affect the fit/validation arrays or the feature shape.
    rows.extend(
        [
            (1, base, 1, "test"),
            (1, base + timedelta(days=13), 0, "test"),
        ]
    )
    duckdb.sql(
        """
        COPY (SELECT * FROM (VALUES
            %s
        ) AS t(merchant_id, event_timestamp, label, split))
        TO ? (FORMAT PARQUET)
        """
        % ",".join(
            "(%d, TIMESTAMPTZ '%s', %d, '%s')"
            % (merchant, timestamp.isoformat(), label, split)
            for merchant, timestamp, label, split in rows
        ),
        params=[str(path)],
    )


def test_fitness_uses_ordered_training_rows_and_excludes_test_rows(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    write_temporal_dataset(dataset_path)

    base = 1_704_067_200_000_000
    day = 86_400_000_000
    columns = {
        "merchant_id": np.ones(12, dtype=np.int64),
        "created_at": base + np.arange(12, dtype=np.int64) * day,
        "amount": np.arange(12, dtype=np.float64) % 2,
    }
    materializer = FeatureMaterializer(columns)
    evaluator = LogisticRegressionFitness(materializer, dataset_path)
    spec = FeatureSpec(Aggregation.MEAN, "amount", RowWindow(1))

    assert evaluator.score_metric == "roc_auc"
    evaluator.prepare_population([spec])

    # Every event sees exactly one preceding transaction, and the two test
    # events are absent.
    np.testing.assert_allclose(
        evaluator._values_for(spec),
        [0.0, 1.0] * 6,
        equal_nan=True,
    )
    assert evaluator.labels.tolist() == [0, 1] * 6
    assert evaluator.fit_indices.tolist() == list(range(9))
    assert evaluator.validation_indices.tolist() == [9, 10, 11]
    assert evaluator(spec) == 1.0


def test_fitness_rejects_a_temporal_fit_without_two_classes(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    duckdb.sql(
        """
        COPY (
            SELECT * FROM (VALUES
                (1, TIMESTAMP '2024-01-01', 0, 'train'),
                (1, TIMESTAMP '2024-01-02', 0, 'train'),
                (1, TIMESTAMP '2024-01-03', 1, 'train')
            ) AS t(merchant_id, event_timestamp, label, split)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(dataset_path)],
    )
    materializer = FeatureMaterializer(
        {
            "merchant_id": np.array([1, 1], dtype=np.int64),
            "created_at": np.array([1, 2], dtype=np.int64),
            "amount": np.array([1.0, 2.0]),
        }
    )

    with pytest.raises(ValueError, match="at least two target classes"):
        LogisticRegressionFitness(materializer, dataset_path)

from datetime import datetime, timedelta, timezone

import duckdb
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

from automatedfe.features import (
    FeatureMaterializer,
    ResidualEvaluator,
    TxFeature,
)
from automatedfe.features.fitness import (
    MIN_LOGIT_WEIGHT,
    RandomForestFitness,
    objectives_are_finite,
)


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
    evaluator = RandomForestFitness(materializer, dataset_path)
    spec = TxFeature("mean", "amount", 1, "row")

    assert evaluator.score_metric == "roc_auc"
    assert evaluator.n_estimators == 50
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
    assert isinstance(evaluator.last_model, RandomForestClassifier)
    assert all(model.n_estimators == 50 for model in evaluator.last_models)
    assert all(model.n_jobs == -1 for model in evaluator.last_models)
    assert len(evaluator.cv_splits) == 3
    assert evaluator.fold_scores == [1.0, 1.0, 1.0]
    for fit_indices, validation_indices in evaluator.cv_splits:
        assert evaluator.event_timestamps[fit_indices[-1]] < evaluator.event_timestamps[
            validation_indices[0]
        ]
        assert not np.intersect1d(fit_indices, validation_indices).size


def test_time_series_folds_never_split_equal_timestamps(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = [
        (1, base + timedelta(days=day), label, "train")
        for day in range(6)
        for label in (0, 1)
    ]
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
        params=[str(dataset_path)],
    )

    evaluator = RandomForestFitness(
        FeatureMaterializer(
            {
                "merchant_id": np.ones(12, dtype=np.int64),
                "created_at": np.repeat(
                    np.arange(6, dtype=np.int64), 2
                ),
                "amount": np.arange(12, dtype=np.float64),
            }
        ),
        dataset_path,
    )

    for fit_indices, validation_indices in evaluator.cv_splits:
        fit_timestamps = evaluator.event_timestamps[fit_indices]
        validation_timestamps = evaluator.event_timestamps[validation_indices]
        assert fit_timestamps[-1] < validation_timestamps[0]
        assert set(fit_timestamps).isdisjoint(validation_timestamps)


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
        RandomForestFitness(materializer, dataset_path)


def test_residual_evaluator_scores_brier_improvement_over_each_fold_baseline(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    duckdb.sql(
        """
        COPY (
            SELECT
                1 AS merchant_id,
                TIMESTAMP '2024-01-01' + (index + 1) * INTERVAL '1 day'
                    AS event_timestamp,
                (index % 2)::INTEGER AS label,
                'train' AS split
            FROM range(1200) AS events(index)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(dataset_path)],
    )

    base = 1_704_067_200_000_000
    day = 86_400_000_000
    evaluator = ResidualEvaluator(
        FeatureMaterializer(
            {
                "merchant_id": np.ones(1200, dtype=np.int64),
                "created_at": base + np.arange(1200, dtype=np.int64) * day,
                "amount": np.arange(1200, dtype=np.float64) % 2,
            }
        ),
        dataset_path,
    )
    feature = TxFeature("mean", "amount", 1, "row")

    score = evaluator(feature)

    assert evaluator.score_metric == "brier_improvement"
    assert score > 0.0
    assert len(evaluator.fold_scores) == 3
    assert isinstance(evaluator.last_model, DecisionTreeRegressor)
    assert hasattr(evaluator.last_model, "min_samples_leaf")
    assert evaluator.fold_scores == pytest.approx(
        [
            1.0 - corrected / baseline
            for baseline, corrected in zip(
                evaluator.fold_baseline_brier_scores,
                evaluator.fold_corrected_brier_scores,
            )
        ]
    )
    assert evaluator.last_model is evaluator.last_models[-1]
    values = evaluator._values_for(feature).reshape(-1, 1)
    for fold, (fit_indices, validation_indices) in enumerate(evaluator.cv_splits):
        assert evaluator.event_timestamps[fit_indices[-1]] < evaluator.event_timestamps[
            validation_indices[0]
        ]
        baseline = evaluator.fold_baselines[fold]
        expected_weight = max(
            baseline * (1.0 - baseline),
            MIN_LOGIT_WEIGHT,
        )
        expected_residuals = (evaluator.labels[fit_indices] - baseline) / expected_weight
        np.testing.assert_allclose(
            evaluator.last_training_weights[fold],
            expected_weight,
        )
        np.testing.assert_allclose(
            evaluator.last_training_residuals[fold],
            expected_residuals,
        )
        model = evaluator.last_models[fold]
        assert model is not None
        baseline_score = np.log(baseline / (1.0 - baseline))
        correction = model.predict(values[validation_indices])
        expected_predictions = 1.0 / (
            1.0 + np.exp(-(baseline_score + 0.2 * correction))
        )
        np.testing.assert_allclose(
            evaluator.last_validation_predictions[fold],
            expected_predictions,
        )


def test_residual_evaluator_gives_a_constant_signal_zero_improvement(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    duckdb.sql(
        """
        COPY (
            SELECT
                1 AS merchant_id,
                TIMESTAMP '2024-01-01' + (index + 1) * INTERVAL '1 day'
                    AS event_timestamp,
                (index % 2)::INTEGER AS label,
                'train' AS split
            FROM range(20) AS events(index)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(dataset_path)],
    )

    class ConstantMaterializer:
        def materialize_for_events(self, _individual, merchants, _timestamps):
            return np.ones(len(merchants), dtype=np.float64)

    evaluator = ResidualEvaluator(ConstantMaterializer(), dataset_path)

    assert evaluator("constant") == pytest.approx(0.0)
    assert evaluator.last_models == [None, None, None]
    for baseline, predictions in zip(
        evaluator.fold_baselines,
        evaluator.last_validation_predictions,
    ):
        np.testing.assert_allclose(predictions, baseline)


def test_residual_evaluator_allows_a_one_class_early_temporal_fold(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    duckdb.sql(
        """
        COPY (
            SELECT * FROM (VALUES
                (1, TIMESTAMP '2024-01-01', 0, 'train'),
                (1, TIMESTAMP '2024-01-02', 0, 'train'),
                (1, TIMESTAMP '2024-01-03', 0, 'train'),
                (1, TIMESTAMP '2024-01-04', 0, 'train'),
                (1, TIMESTAMP '2024-01-05', 1, 'train')
            ) AS t(merchant_id, event_timestamp, label, split)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(dataset_path)],
    )
    evaluator = ResidualEvaluator(
        FeatureMaterializer(
            {
                "merchant_id": np.ones(5, dtype=np.int64),
                "created_at": np.arange(5, dtype=np.int64),
                "amount": np.arange(5, dtype=np.float64),
            }
        ),
        dataset_path,
        n_splits=2,
    )

    score = evaluator(TxFeature("mean", "amount", 1, "row"))

    assert np.isfinite(score)
    assert len(evaluator.fold_scores) == 2


def test_residual_evaluator_rejects_a_globally_one_class_dataset(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    duckdb.sql(
        """
        COPY (
            SELECT
                1 AS merchant_id,
                TIMESTAMP '2024-01-01' + index * INTERVAL '1 day'
                    AS event_timestamp,
                0 AS label,
                'train' AS split
            FROM range(5) AS events(index)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(dataset_path)],
    )

    with pytest.raises(ValueError, match="at least two target classes"):
        ResidualEvaluator(
            FeatureMaterializer(
                {
                    "merchant_id": np.ones(5, dtype=np.int64),
                    "created_at": np.arange(5, dtype=np.int64),
                    "amount": np.arange(5, dtype=np.float64),
                }
            ),
            dataset_path,
            n_splits=2,
        )


def test_objective_vector_returns_fold_scores_and_materialization_time(tmp_path):
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
    evaluator = RandomForestFitness(materializer, dataset_path)
    spec = TxFeature("mean", "amount", 1, "row")
    evaluator.prepare_population([spec])

    vector = evaluator.objective_vector(spec)

    assert len(vector) == 4
    assert vector[:3] == [1.0, 1.0, 1.0]
    assert vector[3] >= 0.0
    assert evaluator.fold_scores == [1.0, 1.0, 1.0]
    # Cache hits reuse the identical materialization duration.
    assert evaluator.objective_vector(spec)[3] == vector[3]
    # The scalar API stays the mean of the fold scores.
    assert evaluator(spec) == float(np.mean(vector[:3]))


def test_residual_objective_vector_returns_fold_scores_and_materialization_time(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    duckdb.sql(
        """
        COPY (
            SELECT
                1 AS merchant_id,
                TIMESTAMP '2024-01-01' + (index + 1) * INTERVAL '1 day'
                    AS event_timestamp,
                (index % 2)::INTEGER AS label,
                'train' AS split
            FROM range(1200) AS events(index)
        ) TO ? (FORMAT PARQUET)
        """,
        params=[str(dataset_path)],
    )

    base = 1_704_067_200_000_000
    day = 86_400_000_000
    materializer = FeatureMaterializer(
        {
            "merchant_id": np.ones(1200, dtype=np.int64),
            "created_at": base + np.arange(1200, dtype=np.int64) * day,
            "amount": np.arange(1200, dtype=np.float64) % 2,
        }
    )
    evaluator = ResidualEvaluator(materializer, dataset_path)
    feature = TxFeature("mean", "amount", 1, "row")

    vector = evaluator.objective_vector(feature)

    assert len(vector) == 4
    assert vector[:3] == pytest.approx(evaluator.fold_scores)
    assert vector[3] >= 0.0
    assert isinstance(evaluator.last_model, DecisionTreeRegressor)
    assert evaluator.objective_vector(feature)[3] == vector[3]
    assert evaluator(feature) == pytest.approx(float(np.mean(vector[:3])))


def test_objective_vector_raises_on_single_class_validation_fold(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    duckdb.sql(
        """
        COPY (SELECT * FROM (VALUES
            (1, TIMESTAMP '2024-01-01', 0, 'train'),
            (1, TIMESTAMP '2024-01-02', 1, 'train'),
            (1, TIMESTAMP '2024-01-03', 0, 'train'),
            (1, TIMESTAMP '2024-01-04', 0, 'train'),
            (1, TIMESTAMP '2024-01-05', 1, 'train')
        ) AS t(merchant_id, event_timestamp, label, split))
        TO ? (FORMAT PARQUET)
        """,
        params=[str(dataset_path)],
    )
    evaluator = RandomForestFitness(
        FeatureMaterializer(
            {
                "merchant_id": np.ones(5, dtype=np.int64),
                "created_at": np.arange(5, dtype=np.int64),
                "amount": np.ones(5, dtype=np.float64),
            }
        ),
        dataset_path,
        n_splits=2,
    )

    with pytest.raises(ValueError, match="ROC AUC requires both target classes"):
        evaluator.objective_vector(TxFeature("mean", "amount", 1, "row"))


def test_objectives_are_finite_distinguishes_invalid_vectors():
    assert objectives_are_finite([0.5, 0.6, 0.4, 0.01])
    assert not objectives_are_finite([0.5, np.nan, 0.4, 0.01])
    assert not objectives_are_finite([0.5, np.inf, 0.4, 0.01])
    assert not objectives_are_finite([])

from datetime import datetime, timedelta, timezone

import duckdb
import numpy as np
import polars as pl
import pytest
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.problems import Fitness, MultiObjectiveProblem
from geneticengine.solutions.individual import ConcreteIndividual

from automatedfe.features import (
    ArchiveSnapshot,
    ArchiveStep,
    FeatureMaterializer,
    FinalEvaluator,
    MeanAmount,
    TotalAmount,
    TxFeature,
)
from automatedfe.features.final_evaluation import FinalEvaluationResult

LABEL_MAPPING = {
    "status": {"approved": 0, "complete": 1},
    "capture_method": {"contactless": 0},
    "payment_method": {"credit": 0},
    "card_brand": {"visa": 0},
    "document_type": {"cpf": 0},
}


def write_train_test_dataset(path, *, uniform_test_label=None):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    # One transaction per day (amounts 0, 1, 0, 1, ...) and one event per
    # day.  An event on day k sees exactly the transaction of day k-1, so
    # the "mean amount, row window 1" feature equals the previous day's
    # amount.  The label is chosen to match that feature.
    for day in range(1, 13):
        label = (day - 1) % 2
        rows.append((1, base + timedelta(days=day), label, "train"))
    # Test events interleave the remaining days and see different
    # transaction histories, producing both feature values.
    for day, label in ((5, 0), (6, 1), (7, 0), (8, 1)):
        if uniform_test_label is not None:
            label = uniform_test_label
        rows.append((1, base + timedelta(days=day), label, "test"))
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


def build_materializer(n_transactions):
    base = 1_704_067_200_000_000
    day = 86_400_000_000
    return FeatureMaterializer(
        {
            "merchant_id": np.ones(n_transactions, dtype=np.int64),
            "created_at": base + np.arange(n_transactions, dtype=np.int64) * day,
            "amount": np.arange(n_transactions, dtype=np.float64) % 2,
        }
    )


def build_archive(expressions):
    scores = {
        str(expression): (
            (0.8, 0.8, 0.8, 1.0)
            if index == 0
            else (0.9, 0.7, 0.9, 2.0)
        )
        for index, expression in enumerate(expressions)
    }
    problem = MultiObjectiveProblem(
        fitness_function=lambda expression: scores[str(expression)],
        minimize=[False, False, False, True],
    )
    individuals = []
    for expression in expressions:
        individual = ConcreteIndividual(expression)
        individual.set_fitness(problem, Fitness(scores[str(expression)]))
        individuals.append(individual)

    archive = ArchiveStep(mapping=LABEL_MAPPING)
    list(
        archive.apply(
            problem,
            SequentialEvaluator(),
            representation=None,
            random=None,
            population=iter(individuals),
            target_size=len(individuals),
            generation=0,
        )
    )
    return archive


def test_final_evaluator_fits_train_and_scores_test(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    write_train_test_dataset(dataset_path)

    evaluator = FinalEvaluator(build_materializer(12), dataset_path)
    feature = TxFeature("mean", "amount", 1, "row")

    result = evaluator.evaluate([feature])

    assert isinstance(result, FinalEvaluationResult)
    # 12 train rows followed by 4 test rows, all materialized.
    assert len(evaluator.event_merchants) == 16
    assert evaluator.train_indices.tolist() == list(range(12))
    assert evaluator.test_indices.tolist() == list(range(12, 16))
    # The training events see the same previous-day amount as in
    # test_fitness_uses_ordered_training_rows_and_excludes_test_rows.
    np.testing.assert_allclose(
        evaluator.materializer.materialize_for_events(
            feature, evaluator.event_merchants, evaluator.event_timestamps
        ),
        [0.0, 1.0] * 6 + [0.0, 1.0, 0.0, 1.0],
        equal_nan=True,
    )
    assert result.metrics == {"accuracy": 1.0, "roc_auc": 1.0}
    assert result.predictions.tolist() == [0, 1, 0, 1]
    assert result.model.coef_.shape == (1, 1)


def test_final_evaluator_builds_matrix_from_expressions_and_deduplicates(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    write_train_test_dataset(dataset_path)

    evaluator = FinalEvaluator(build_materializer(12), dataset_path)
    individuals = [
        MeanAmount(1),
        TotalAmount(2),
        MeanAmount(1),
    ]

    result = evaluator.evaluate(individuals)

    assert result.model.coef_.shape == (1, 2)
    assert result.metrics["accuracy"] >= 0.0


def test_final_evaluator_rejects_empty_feature_set(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    write_train_test_dataset(dataset_path)

    evaluator = FinalEvaluator(build_materializer(12), dataset_path)

    with pytest.raises(ValueError, match="At least one individual"):
        evaluator.evaluate([])


def test_final_evaluator_requires_two_classes_in_test_split(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    write_train_test_dataset(dataset_path, uniform_test_label=1)

    evaluator = FinalEvaluator(build_materializer(12), dataset_path)

    with pytest.raises(ValueError, match="both target classes in the test split"):
        evaluator.evaluate([TxFeature("mean", "amount", 1, "row")])


def test_final_evaluator_rejects_missing_train_or_test_rows(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    write_train_test_dataset(dataset_path)
    train_only = tmp_path / "train_only.parquet"
    pl.read_parquet(dataset_path).filter(
        pl.col("split") != "test"
    ).write_parquet(train_only)

    with pytest.raises(ValueError, match="no rows in the test split"):
        FinalEvaluator(build_materializer(12), train_only)


def test_final_evaluator_accepts_live_and_loaded_archives_in_stable_order(
    tmp_path,
    monkeypatch,
):
    dataset_path = tmp_path / "dataset.parquet"
    write_train_test_dataset(dataset_path)
    expressions = [MeanAmount(1), TotalAmount(2)]
    live_archive = build_archive(expressions)
    archive_path = tmp_path / "archive.json"
    live_archive.save(archive_path)

    evaluator = FinalEvaluator(build_materializer(12), dataset_path)
    materialized = []
    original = evaluator.materializer.materialize_for_events

    def recording_materialize(individual, event_merchants, event_timestamps):
        materialized.append(str(individual))
        return original(individual, event_merchants, event_timestamps)

    monkeypatch.setattr(
        evaluator.materializer,
        "materialize_for_events",
        recording_materialize,
    )
    live_result = evaluator.evaluate(live_archive)
    loaded_result = evaluator.evaluate(archive_path)

    assert materialized[:2] == [str(expression) for expression in expressions]
    assert materialized[2:4] == [str(expression) for expression in expressions]
    assert loaded_result.metrics == live_result.metrics
    np.testing.assert_array_equal(loaded_result.predictions, live_result.predictions)
    np.testing.assert_allclose(loaded_result.model.coef_, live_result.model.coef_)


def test_final_evaluator_can_be_configured_with_an_archive(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    write_train_test_dataset(dataset_path)
    archive = ArchiveSnapshot(
        version=1,
        minimize=(False, False, False, True),
        mapping=LABEL_MAPPING,
        expressions=(MeanAmount(1),),
        objectives=((0.8, 0.8, 0.8, 1.0),),
    )

    result = FinalEvaluator(
        build_materializer(12),
        dataset_path,
        archive=archive,
    ).evaluate()

    assert result.model.n_features_in_ == 1


def test_final_evaluator_rejects_empty_archive(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    write_train_test_dataset(dataset_path)
    archive = ArchiveSnapshot(
        version=1,
        minimize=(False, False, False, True),
        mapping=LABEL_MAPPING,
        expressions=(),
        objectives=(),
    )

    evaluator = FinalEvaluator(build_materializer(12), dataset_path)
    with pytest.raises(ValueError, match="Archive is empty"):
        evaluator.evaluate(archive)


def test_final_evaluator_rejects_incompatible_archive_configuration(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    write_train_test_dataset(dataset_path)
    archive = ArchiveSnapshot(
        version=1,
        minimize=(True, True, True, True),
        mapping=LABEL_MAPPING,
        expressions=(MeanAmount(1),),
        objectives=((0.8, 0.8, 0.8, 1.0),),
    )

    evaluator = FinalEvaluator(build_materializer(12), dataset_path)
    with pytest.raises(ValueError, match="objective directions"):
        evaluator.evaluate(archive)


def test_final_evaluator_validates_loaded_archive_mapping(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    write_train_test_dataset(dataset_path)
    archive_path = tmp_path / "archive.json"
    build_archive([MeanAmount(1)]).save(archive_path)
    incompatible_mapping = {
        **LABEL_MAPPING,
        "status": {"approved": 0, "complete": 7},
    }

    evaluator = FinalEvaluator(
        build_materializer(12),
        dataset_path,
        mapping=incompatible_mapping,
    )
    with pytest.raises(ValueError, match="label mapping is incompatible"):
        evaluator.evaluate(archive_path)

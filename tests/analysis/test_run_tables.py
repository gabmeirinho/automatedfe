import numpy as np

from automatedfe.analysis.run_tables import (
    build_final_evaluation_tables,
    read_final_evaluation_tables,
    write_final_evaluation_tables,
)
from automatedfe.evaluation.final_evaluation import FinalEvaluator
from automatedfe.features.grammar import MeanAmount, TotalAmount

from tests.evaluation.test_final_evaluation import (
    build_materializer,
    write_train_test_dataset,
)


def test_final_evaluation_tables_are_complete_and_model_free(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    write_train_test_dataset(dataset_path)
    expressions = [MeanAmount(1), TotalAmount(2), MeanAmount(1)]
    evaluator = FinalEvaluator(
        build_materializer(12),
        dataset_path,
        n_estimators=9,
        max_samples=None,
        n_jobs=1,
        random_state=17,
    )

    result = evaluator.evaluate(
        expressions,
        search_fold_scores=((0.1, 0.2, 0.3, 0.01), (0.4, 0.5, 0.6, 0.02)),
    )
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.feature_count == 2
    assert len(set(diagnostics.feature_ids)) == 2
    assert diagnostics.correlation_training_row_count == 12
    assert diagnostics.tree_importances.shape == (9, 2)
    np.testing.assert_allclose(
        diagnostics.importance_means,
        diagnostics.tree_importances.mean(axis=0),
    )
    np.testing.assert_allclose(
        diagnostics.importance_stds,
        diagnostics.tree_importances.std(axis=0),
    )
    assert diagnostics.total_materialization_seconds == sum(
        diagnostics.materialization_seconds
    )

    tables = build_final_evaluation_tables(result)
    assert len(tables.features) == 2
    assert len(tables.importances) == 2
    assert len(tables.correlations) == 4
    assert tables.metrics["correlation_training_row_count"] == 12
    assert "accuracy" not in tables.metrics

    paths = write_final_evaluation_tables(tmp_path, tables)
    reopened = read_final_evaluation_tables(tmp_path, paths)
    assert reopened.metrics["roc_auc"] == tables.metrics["roc_auc"]
    assert len(reopened.features) == 2
    assert len(reopened.correlations) == 4
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / value for value in paths.values())
    )
    assert "model" not in persisted
    assert "predictions" not in persisted
    assert "accuracy" not in persisted


def test_spearman_uses_average_ranks_for_ties():
    values = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [2.0, 30.0],
            [4.0, 40.0],
        ]
    )
    matrix = FinalEvaluator._spearman_matrix(values)
    assert matrix[0, 1] > 0.9
    np.testing.assert_allclose(matrix, matrix.T)

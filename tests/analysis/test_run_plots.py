from pathlib import Path

import pandas as pd
from PIL import Image

from automatedfe.analysis.artifacts import (
    CANDIDATES_COLUMNS,
    GENERATIONS_COLUMNS,
    write_candidates_csv,
    write_generations_csv,
)
from automatedfe.analysis.run_plots import (
    FIGURE_DPI,
    FIGURE_FILENAMES,
    fold_stability,
    generation_iqr,
    metric_display_name,
    render_run_figures,
)
from automatedfe.analysis.run_tables import (
    FinalEvaluationTables,
    read_final_evaluation_tables,
    write_final_evaluation_tables,
)


def _row(columns, **values):
    row = {column: "" for column in columns}
    row.update(values)
    return row


def _persist_plot_tables(run_dir: Path, *, evaluation_free: bool = False) -> None:
    candidates = []
    for index in range(6):
        candidates.append(
            _row(
                CANDIDATES_COLUMNS,
                Strategy=(
                    "enumerative_without_archive" if evaluation_free else "enumerative"
                ),
                CandidateIndex=index,
                Generation=index // 3,
                Expression=f"feat_mean_amount_row_{index + 1}",
                Dependencies=f"feat_mean_amount_row_{index + 1}",
                Split1="" if evaluation_free else 0.1 + index * 0.01,
                Split2="" if evaluation_free else 0.2 + index * 0.01,
                Split3="" if evaluation_free else 0.3 + index * 0.01,
                MaterializationTime="" if evaluation_free else index + 1,
                ArchiveMember=False if evaluation_free else index < 3,
                Status="generated" if evaluation_free else "evaluated",
            )
        )
    generations = [
        _row(
            GENERATIONS_COLUMNS,
            Strategy="enumerative",
            Generation=generation,
            Generated=3,
            Unique=3,
            Duplicate=0,
            Invalid=0,
            Evaluated=0 if evaluation_free else 3,
            ArchiveSize=0 if evaluation_free else generation + 2,
            Added=0 if evaluation_free else 2,
            DurationSeconds=1.5,
            CumulativeRuntimeSeconds=(generation + 1) * 1.5,
        )
        for generation in range(2)
    ]
    write_candidates_csv(run_dir / "candidates.csv", candidates)
    write_generations_csv(run_dir / "generations.csv", generations)

    count = 21
    feature_rows = tuple(
        {
            "feature_id": f"feat_{index:02d}",
            "feature_label": f"feat_mean_amount_row_{index + 1}",
            "search_fold_1": None if evaluation_free else 0.1 + index / 100,
            "search_fold_2": None if evaluation_free else 0.2 + index / 100,
            "search_fold_3": None if evaluation_free else 0.3 + index / 100,
        }
        for index in range(count)
    )
    importance_rows = tuple(
        {
            "feature_id": f"feat_{index:02d}",
            "feature_label": f"feat_mean_amount_row_{index + 1}",
            "importance_mean": (index + 1) / sum(range(1, count + 1)),
            "importance_std": 0.001 * index,
        }
        for index in range(count)
    )
    correlation_rows = tuple(
        {
            "feature_id": f"feat_{row:02d}",
            "feature_label": f"feat_mean_amount_row_{row + 1}",
            "other_feature_id": f"feat_{column:02d}",
            "other_feature_label": f"feat_mean_amount_row_{column + 1}",
            "spearman": 1.0 if row == column else 0.05,
            "training_row_count": 37,
        }
        for row in range(count)
        for column in range(count)
    )
    timing_rows = tuple(
        {
            "feature_id": f"feat_{index:02d}",
            "feature_label": f"feat_mean_amount_row_{index + 1}",
            "materialization_seconds": index / 10,
        }
        for index in range(count)
    )
    write_final_evaluation_tables(
        run_dir,
        FinalEvaluationTables(
            features=feature_rows,
            metrics={
                "roc_auc": 0.81,
                "search_fold_metric": "brier_improvement",
                "correlation_training_row_count": 37,
                "total_materialization_seconds": sum(
                    index / 10 for index in range(count)
                ),
            },
            importances=importance_rows,
            correlations=correlation_rows,
            timings=timing_rows,
        ),
    )


def test_fold_stability_and_generation_iqr_are_exact():
    frame = pd.DataFrame(
        {
            "Generation": [0, 0, 0, 1],
            "Split1": [1.0, 2.0, 3.0, 4.0],
            "Split2": [2.0, 2.0, 3.0, 5.0],
            "Split3": [3.0, 2.0, 3.0, 6.0],
        }
    )
    stability = fold_stability(frame)
    assert stability.round(6).tolist() == [0.816497, 0.0, 0.0, 0.816497]
    frame["stability"] = stability
    summary = generation_iqr(frame, "stability")
    first = summary.iloc[0]
    assert first["median"] == 0.0
    assert first["q1"] == 0.0
    assert first["q3"] == stability.iloc[0] / 2


def test_all_eleven_figures_are_table_driven_and_300_dpi(tmp_path):
    _persist_plot_tables(tmp_path)

    paths = render_run_figures(tmp_path)

    assert tuple(path.name for path in paths) == FIGURE_FILENAMES
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    for path in paths:
        with Image.open(path) as image:
            assert round(image.info["dpi"][0]) == FIGURE_DPI
            assert round(image.info["dpi"][1]) == FIGURE_DPI
    # Figure limits do not truncate either complete feature-level CSV.
    reopened = read_final_evaluation_tables(tmp_path)
    assert len(reopened.importances) == 21
    assert len(reopened.correlations) == 21 * 21


def test_evaluation_free_run_renders_explicit_panels_without_scores(tmp_path):
    _persist_plot_tables(tmp_path, evaluation_free=True)
    paths = render_run_figures(tmp_path)
    assert len(paths) == 11
    assert all(path.is_file() for path in paths)


def test_metric_labels_do_not_misname_brier_improvement():
    assert metric_display_name("brier") == "Brier improvement"
    assert metric_display_name("brier_improvement") == "Brier improvement"
    assert metric_display_name("roc_auc") == "ROC AUC"

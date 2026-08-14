import hashlib
import json
from pathlib import Path

import pytest

import automatedfe.analysis.run_report as report_module
from automatedfe.analysis.artifacts import CANDIDATES_COLUMNS, GENERATIONS_COLUMNS
from automatedfe.analysis.run_bundle import RunBundleWriter
from automatedfe.analysis.run_plots import FIGURE_FILENAMES
from automatedfe.analysis.run_report import (
    REPORT_FORMAT,
    REPORT_SCHEMA_VERSION,
    render_run_report,
)
from automatedfe.analysis.run_tables import FinalEvaluationTables
from automatedfe.features.grammar import MeanAmount, TotalAmount
from automatedfe.search.archive import build_snapshot_document


MAPPING = {
    "status": {"approved": 0},
    "capture_method": {"contactless": 0},
    "payment_method": {"credit": 0},
    "card_brand": {"visa": 0},
    "document_type": {"cpf": 0},
}


def _row(columns, **values):
    row = {column: "" for column in columns}
    row.update(values)
    return row


def _tables() -> FinalEvaluationTables:
    features = (
        {
            "feature_id": "feat_mean",
            "feature_label": "feat_mean_amount_row_5",
            "search_fold_1": 0.11,
            "search_fold_2": 0.14,
            "search_fold_3": 0.13,
        },
        {
            "feature_id": "feat_total",
            "feature_label": "feat_total_amount_row_10",
            "search_fold_1": 0.18,
            "search_fold_2": 0.16,
            "search_fold_3": 0.17,
        },
    )
    importances = (
        {
            "feature_id": "feat_mean",
            "feature_label": "feat_mean_amount_row_5",
            "importance_mean": 0.35,
            "importance_std": 0.04,
        },
        {
            "feature_id": "feat_total",
            "feature_label": "feat_total_amount_row_10",
            "importance_mean": 0.65,
            "importance_std": 0.06,
        },
    )
    correlations = tuple(
        {
            "feature_id": feature_id,
            "feature_label": feature_label,
            "other_feature_id": other_id,
            "other_feature_label": other_label,
            "spearman": 1.0 if feature_id == other_id else 0.27,
            "training_row_count": 48,
        }
        for feature_id, feature_label in (
            ("feat_mean", "feat_mean_amount_row_5"),
            ("feat_total", "feat_total_amount_row_10"),
        )
        for other_id, other_label in (
            ("feat_mean", "feat_mean_amount_row_5"),
            ("feat_total", "feat_total_amount_row_10"),
        )
    )
    timings = (
        {
            "feature_id": "feat_mean",
            "feature_label": "feat_mean_amount_row_5",
            "materialization_seconds": 0.7,
        },
        {
            "feature_id": "feat_total",
            "feature_label": "feat_total_amount_row_10",
            "materialization_seconds": 1.1,
        },
    )
    return FinalEvaluationTables(
        features=features,
        metrics={
            "roc_auc": 0.8125,
            "search_fold_metric": "brier_improvement",
            "correlation_training_row_count": 48,
            "total_materialization_seconds": 1.8,
        },
        importances=importances,
        correlations=correlations,
        timings=timings,
    )


def _run_bundle(
    tmp_path: Path,
    *,
    strategy: str = "enumerative",
    configuration: dict[str, object] | None = None,
) -> Path:
    dataset = tmp_path / "dataset.parquet"
    dataset.write_bytes(b"persisted dataset fingerprint source")
    mmap_dir = tmp_path / "mmap"
    mmap_dir.mkdir()
    (mmap_dir / "manifest.json").write_text(
        json.dumps({"rows": 48, "columns": {}}), encoding="utf-8"
    )
    expressions = (MeanAmount(0), TotalAmount(1))
    objectives = ((0.11, 0.14, 0.13, 0.7), (0.18, 0.16, 0.17, 1.1))
    snapshot = build_snapshot_document(
        expressions,
        objectives,
        minimize=[False, False, False, True],
        mapping_ref={"file": "manifest.json", "source": "run_manifest"},
    )
    candidates = [
        _row(
            CANDIDATES_COLUMNS,
            Strategy="enumerative",
            CandidateIndex=index,
            Generation=index,
            Expression=str(expression),
            Dependencies=str(expression),
            Split1=scores[0],
            Split2=scores[1],
            Split3=scores[2],
            MaterializationTime=scores[3],
            ArchiveMember=True,
            Status="evaluated",
        )
        for index, (expression, scores) in enumerate(zip(expressions, objectives))
    ]
    generations = [
        _row(
            GENERATIONS_COLUMNS,
            Strategy="enumerative",
            Generation=generation,
            Generated=1,
            Unique=1,
            Duplicate=0,
            Invalid=0,
            Evaluated=1,
            ArchiveSize=generation + 1,
            Added=1,
            DurationSeconds=1.2,
            CumulativeRuntimeSeconds=(generation + 1) * 1.2,
        )
        for generation in range(2)
    ]

    class Lifecycle:
        candidate_rows = candidates
        generation_rows = generations
        snapshot_documents = ((0, snapshot), (1, snapshot))

    writer = RunBundleWriter(
        tmp_path / "run",
        run_id="thesis-seed-07",
        strategy=strategy,
        dataset_path=dataset,
        mapping=MAPPING,
        mmap_dir=mmap_dir,
        configuration=configuration
        or {"time_budget_seconds": 120.0, "candidate_count": None},
        created_at_utc="2026-08-14T09:30:00Z",
    )
    writer.write_evaluation_tables(_tables())
    return writer.finalize("search_complete", lifecycle=Lifecycle()).path


@pytest.mark.parametrize("feature_labels", ["id", "expression"])
def test_report_contains_figures_tables_metadata_and_caveats(tmp_path, feature_labels):
    run_dir = _run_bundle(tmp_path)

    report = render_run_report(
        run_dir,
        feature_labels=feature_labels,
        rendered_at_utc="2026-08-14T12:00:00Z",
    )

    document = report.read_text(encoding="utf-8")
    versions = list((run_dir / "report-artifacts").iterdir())
    assert len(versions) == 1
    metadata = json.loads((versions[0] / "report.json").read_text(encoding="utf-8"))
    asset_prefix = f"report-artifacts/{versions[0].name}"
    assert report == run_dir / "report.html"
    assert metadata == {
        "feature_labels": feature_labels,
        "figures": [f"{asset_prefix}/figures/{name}" for name in FIGURE_FILENAMES],
        "format": REPORT_FORMAT,
        "report": "report.html",
        "rerendered_at_utc": "2026-08-14T12:00:00Z",
        "schema_version": REPORT_SCHEMA_VERSION,
        "tables": [
            "candidates.csv",
            "generations.csv",
            "evaluation/features.csv",
            "evaluation/importances.csv",
            "evaluation/correlations.csv",
            "evaluation/timings.csv",
        ],
    }
    assert document.count("<figure id=") == 11
    assert all(
        f"{asset_prefix}/figures/{name}" in document for name in FIGURE_FILENAMES
    )
    assert all(path in document for path in metadata["tables"])
    assert f'content="{feature_labels}"' in document
    assert "Brier improvement is never relabeled as ROC AUC" in document
    assert "complete imputed training split (48 rows)" in document
    assert "cache read can be faster" in document
    assert "independent optimization objectives" in document
    assert "Held-out ROC AUC" in document
    assert "<th>Search budget</th><td>120 seconds</td>" in document
    assert "<th>Budget basis</th><td>Wall-clock search time</td>" in document
    assert "accuracy" in document
    assert "file://" not in document


def test_report_identifies_candidate_budget_without_time_budget(tmp_path):
    run_dir = _run_bundle(
        tmp_path,
        strategy="enumerative_without_archive",
        configuration={"time_budget_seconds": None, "candidate_count": 2500},
    )

    document = render_run_report(run_dir).read_text(encoding="utf-8")

    assert "<th>Search budget</th><td>2,500 candidates</td>" in document
    assert (
        "<th>Budget basis</th><td>Candidate count · enumerative without archive "
        "(no time budget)</td>" in document
    )


def test_rerender_reads_only_persisted_inputs_and_preserves_nonreport_artifacts(
    tmp_path, monkeypatch
):
    run_dir = _run_bundle(tmp_path)
    protected = [
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in {"report.html", "report.json"}
    ]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}

    from automatedfe.evaluation.final_evaluation import FinalEvaluator
    from automatedfe.features.feature_materialization import FeatureMaterializer

    monkeypatch.setattr(
        FinalEvaluator,
        "evaluate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("evaluation must not run")
        ),
    )
    monkeypatch.setattr(
        FeatureMaterializer,
        "materialize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("materialization must not run")
        ),
    )
    render_run_report(run_dir, feature_labels="id")

    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in before.items()
    )
    assert not list(run_dir.glob(".report.*"))


def test_render_failure_preserves_previous_report(tmp_path, monkeypatch):
    run_dir = _run_bundle(tmp_path)
    render_run_report(
        run_dir,
        feature_labels="expression",
        rendered_at_utc="2026-08-14T12:00:00Z",
    )
    old_html = (run_dir / "report.html").read_bytes()
    old_version = next((run_dir / "report-artifacts").iterdir())
    old_metadata = (old_version / "report.json").read_bytes()
    old_figures = {
        path.name: path.read_bytes() for path in (old_version / "figures").iterdir()
    }

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("plot rendering failed")

    monkeypatch.setattr(report_module, "render_run_figures", fail_render)
    with pytest.raises(RuntimeError, match="plot rendering failed"):
        render_run_report(run_dir, feature_labels="id")

    assert (run_dir / "report.html").read_bytes() == old_html
    assert (old_version / "report.json").read_bytes() == old_metadata
    assert {
        path.name: path.read_bytes() for path in (old_version / "figures").iterdir()
    } == old_figures
    assert not list(run_dir.glob(".report.*"))
    assert list((run_dir / "report-artifacts").iterdir()) == [old_version]


def test_publication_fault_keeps_previous_committed_report(tmp_path, monkeypatch):
    run_dir = _run_bundle(tmp_path)
    render_run_report(
        run_dir,
        feature_labels="expression",
        rendered_at_utc="2026-08-14T12:00:00Z",
    )
    old_html = (run_dir / "report.html").read_bytes()
    old_versions = {path.name for path in (run_dir / "report-artifacts").iterdir()}
    real_replace = report_module.os.replace

    def fail_commit(source, destination):
        if Path(destination) == run_dir / "report.html":
            raise OSError("commit point failed")
        return real_replace(source, destination)

    monkeypatch.setattr(report_module.os, "replace", fail_commit)
    with pytest.raises(OSError, match="commit point failed"):
        render_run_report(run_dir, feature_labels="id")

    assert (run_dir / "report.html").read_bytes() == old_html
    assert {
        path.name for path in (run_dir / "report-artifacts").iterdir()
    } == old_versions


def test_report_rejects_unknown_label_mode_before_writing(tmp_path):
    run_dir = _run_bundle(tmp_path)
    with pytest.raises(ValueError, match="feature_labels"):
        render_run_report(run_dir, feature_labels="name")
    assert not (run_dir / "report.html").exists()

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import automatedfe.analysis.run_services as service_module
from automatedfe.analysis import (
    RunAnalysisError,
    RunReportError,
    RunStateError,
    analyze_run,
    rerender_tracked_run_report,
)
from automatedfe.analysis.artifacts import (
    fingerprint_mapping,
    fingerprint_mmap_manifest,
)
from automatedfe.analysis.run_bundle import RunBundleWriter
from automatedfe.tracking import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_DATABASE_PATH,
    DEFAULT_TRACKING_URI,
    EXPERIMENT_NAME,
    MlflowStore,
    TrackingStoreError,
    build_run_name,
    resolve_tracking_uri,
)


def _completed_bundle(root: Path, run_id: str) -> Path:
    mapping = {"merchant": {"one": 1}}
    mmap_manifest = {"format": "test-mmap", "version": 1}
    inputs = {
        "dataset": {
            "source_path": None,
            "fingerprint": "sha256:" + "a" * 64,
            "bytes": 0,
        },
        "mapping": {
            "source_path": None,
            "fingerprint": fingerprint_mapping(mapping),
            "mapping": mapping,
        },
        "mmap_manifest": {
            "source_path": None,
            "fingerprint": fingerprint_mmap_manifest(mmap_manifest),
            "manifest": mmap_manifest,
        },
    }
    writer = RunBundleWriter(
        root / "staged-bundle",
        run_id=run_id,
        strategy="enumerative_without_archive",
        inputs=inputs,
        configuration={"time_budget_seconds": None, "candidate_count": 2},
    )
    return writer.finalize("search_complete").path


def _tracked_bundle(
    mlflow_store,
    tmp_path,
    *,
    state,
    with_report=False,
):
    run = mlflow_store.create_run(
        "enumerative_without_archive",
        3,
        parameters={"feature_labels": "expression"},
    )
    staging = _completed_bundle(tmp_path, run.info.run_id)
    if with_report:
        (staging / "report.html").write_text("old report", encoding="utf-8")
        old = staging / "report-artifacts" / "old-version"
        old.mkdir(parents=True)
        (old / "report_metadata.json").write_text("{}", encoding="utf-8")
    mlflow_store.log_artifact_bundle(run.info.run_id, staging)
    mlflow_store.terminate_run(run.info.run_id, state)
    return run.info.run_id


def test_default_configuration_uses_project_results_paths():
    assert DEFAULT_DATABASE_PATH.name == "mlflow.db"
    assert DEFAULT_DATABASE_PATH.parent.name == "results"
    assert DEFAULT_ARTIFACT_ROOT == DEFAULT_DATABASE_PATH.parent / "artifacts"
    assert DEFAULT_TRACKING_URI == f"sqlite:///{DEFAULT_DATABASE_PATH}"
    assert EXPERIMENT_NAME == "automatedfe"


def test_tracking_uri_explicit_override_precedes_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://environment.invalid")
    assert resolve_tracking_uri("sqlite:///explicit.db") == "sqlite:///explicit.db"
    assert resolve_tracking_uri() == "http://environment.invalid"


def test_run_name_is_utc_and_run_ids_make_same_name_collision_safe(mlflow_store):
    instant = datetime(2025, 2, 3, 4, 5, 6, 7, tzinfo=timezone(timedelta(hours=2)))
    expected = "20250203T020506.000007Z_genetic_seed9"
    assert build_run_name("genetic", 9, timestamp=instant) == expected

    first = mlflow_store.create_run("genetic", 9, timestamp=instant)
    second = mlflow_store.create_run("genetic", 9, timestamp=instant)
    assert first.info.run_name == second.info.run_name == expected
    assert first.info.run_id != second.info.run_id


def test_create_log_search_and_read_run_metadata(mlflow_store):
    run = mlflow_store.create_run(
        "genetic",
        17,
        parameters={"time_budget_seconds": 3.5, "feature_labels": ["a", "b"]},
        fingerprints={
            "dataset": {"fingerprint": "sha256:" + "d" * 64},
            "mapping": "sha256:" + "e" * 64,
        },
        strategy_group="gp",
    )
    mlflow_store.log_generation_metrics(
        run.info.run_id, 4, {"generated": 12, "archive_size": 3}
    )
    finished = mlflow_store.terminate_run(run.info.run_id, "complete")

    assert finished.info.status == "FINISHED"
    assert finished.data.params["time_budget_seconds"] == "3.5"
    assert finished.data.params["feature_labels"] == '["a","b"]'
    assert finished.data.tags["strategy_group"] == "gp"
    assert finished.data.tags["fingerprint.dataset"] == "sha256:" + "d" * 64
    assert finished.data.tags["project_state"] == "complete"
    assert finished.data.metrics["archive_size"] == 3
    history = mlflow_store.client.get_metric_history(run.info.run_id, "generated")
    assert [(metric.step, metric.value) for metric in history] == [(4, 12)]
    assert [
        item.info.run_id
        for item in mlflow_store.search_runs(project_states=["complete"])
    ] == [run.info.run_id]


def test_deleted_runs_are_excluded_by_default(mlflow_store):
    run = mlflow_store.create_run("random", 2)
    mlflow_store.delete_run(run.info.run_id)

    assert run.info.run_id not in {
        item.info.run_id for item in mlflow_store.search_runs()
    }
    assert run.info.run_id in {
        item.info.run_id for item in mlflow_store.search_runs(include_deleted=True)
    }
    with pytest.raises(TrackingStoreError, match=run.info.run_id):
        mlflow_store.get_run(run.info.run_id)


def test_artifact_bundle_is_validated_uploaded_and_staging_removed(
    mlflow_store, tmp_path
):
    run = mlflow_store.create_run("enumerative_without_archive", 4)
    staging = _completed_bundle(tmp_path, run.info.run_id)

    mlflow_store.log_artifact_bundle(run.info.run_id, staging)

    assert not staging.exists()
    downloaded = mlflow_store.download_artifact_bundle(
        run.info.run_id, tmp_path / "downloaded"
    )
    assert downloaded.run_id == run.info.run_id
    artifact_names = {
        item.path for item in mlflow_store.client.list_artifacts(run.info.run_id)
    }
    assert "manifest.json" in artifact_names
    assert "MLmodel" not in artifact_names


def test_invalid_bundle_is_not_deleted(mlflow_store, tmp_path):
    run = mlflow_store.create_run("random", 8)
    staging = tmp_path / "bad-staging"
    staging.mkdir()
    (staging / "payload.txt").write_text("not a bundle", encoding="utf-8")

    with pytest.raises(Exception, match="bundle|manifest"):
        mlflow_store.log_artifact_bundle(run.info.run_id, staging)
    assert staging.is_dir()


def test_unavailable_artifact_store_fails_during_construction(tmp_path):
    artifact_file = tmp_path / "not-a-directory"
    artifact_file.write_text("occupied", encoding="utf-8")

    with pytest.raises(TrackingStoreError, match="artifact|directory"):
        MlflowStore(f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=artifact_file)


def test_analysis_retry_updates_original_run_only_after_upload(
    mlflow_store, tmp_path, monkeypatch
):
    run_id = _tracked_bundle(
        mlflow_store, tmp_path, state="analysis_failed"
    )
    rendered = []

    def fake_render(root, *, feature_labels):
        rendered.append((Path(root), feature_labels))
        (Path(root) / "report.html").write_text("recovered", encoding="utf-8")
        return Path(root) / "report.html"

    monkeypatch.setattr(service_module, "render_run_report", fake_render)

    assert analyze_run(run_id, tracking_store=mlflow_store) == run_id
    run = mlflow_store.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["project_state"] == "complete"
    assert rendered[0][1] == "expression"
    assert len(mlflow_store.search_runs()) == 1
    bundle = mlflow_store.download_artifact_bundle(
        run_id, tmp_path / "recovered"
    )
    assert (bundle.path / "report.html").read_text() == "recovered"


def test_repeated_analysis_failure_stays_analysis_failed(
    mlflow_store, tmp_path, monkeypatch
):
    run_id = _tracked_bundle(
        mlflow_store, tmp_path, state="analysis_failed"
    )
    monkeypatch.setattr(
        service_module,
        "render_run_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("still broken")),
    )

    with pytest.raises(RunAnalysisError, match=f"{run_id}.*still broken"):
        analyze_run(run_id, tracking_store=mlflow_store)
    run = mlflow_store.get_run(run_id)
    assert run.info.status == "FAILED"
    assert run.data.tags["project_state"] == "analysis_failed"


def test_report_rerender_updates_same_run_and_preserves_old_version(
    mlflow_store, tmp_path, monkeypatch
):
    run_id = _tracked_bundle(
        mlflow_store, tmp_path, state="complete", with_report=True
    )

    def fake_render(root, *, feature_labels):
        root = Path(root)
        assert feature_labels == "id"
        version = root / "report-artifacts" / "new-version"
        version.mkdir(parents=True)
        (version / "report_metadata.json").write_text("{}", encoding="utf-8")
        (root / "report.html").write_text("new report", encoding="utf-8")
        return root / "report.html"

    monkeypatch.setattr(service_module, "render_run_report", fake_render)

    assert rerender_tracked_run_report(
        run_id, feature_labels="id", tracking_store=mlflow_store
    ) == run_id
    assert len(mlflow_store.search_runs()) == 1
    bundle = mlflow_store.download_artifact_bundle(
        run_id, tmp_path / "rerendered"
    )
    assert (bundle.path / "report.html").read_text() == "new report"
    assert (bundle.path / "report-artifacts/old-version").is_dir()
    assert (bundle.path / "report-artifacts/new-version").is_dir()


def test_failed_rerender_preserves_previous_remote_report(
    mlflow_store, tmp_path, monkeypatch
):
    run_id = _tracked_bundle(
        mlflow_store, tmp_path, state="complete", with_report=True
    )

    def fail_after_local_change(root, **_kwargs):
        (Path(root) / "report.html").write_text("bad candidate", encoding="utf-8")
        raise ValueError("render validation failed")

    monkeypatch.setattr(service_module, "render_run_report", fail_after_local_change)

    with pytest.raises(RunReportError, match=f"{run_id}.*validation failed"):
        rerender_tracked_run_report(run_id, tracking_store=mlflow_store)
    bundle = mlflow_store.download_artifact_bundle(
        run_id, tmp_path / "preserved"
    )
    assert (bundle.path / "report.html").read_text() == "old report"
    assert mlflow_store.get_run(run_id).data.tags["project_state"] == "complete"


def test_recovery_commands_reject_wrong_states(mlflow_store, tmp_path):
    complete = _tracked_bundle(mlflow_store, tmp_path, state="complete")
    with pytest.raises(RunStateError, match=f"{complete}.*analysis_failed"):
        analyze_run(complete, tracking_store=mlflow_store)

    other_root = tmp_path / "other"
    other_root.mkdir()
    failed = _tracked_bundle(
        mlflow_store, other_root, state="analysis_failed"
    )
    with pytest.raises(RunStateError, match=f"{failed}.*FINISHED"):
        rerender_tracked_run_report(failed, tracking_store=mlflow_store)


def test_missing_run_artifacts_name_run_and_do_not_promote(mlflow_store, tmp_path):
    run = mlflow_store.create_run("random", 10)
    mlflow_store.terminate_run(run.info.run_id, "analysis_failed")

    with pytest.raises(RunAnalysisError, match=run.info.run_id):
        analyze_run(run.info.run_id, tracking_store=mlflow_store)
    assert (
        mlflow_store.get_run(run.info.run_id).data.tags["project_state"]
        == "analysis_failed"
    )


def test_corrupt_artifacts_are_rejected_with_run_id(
    mlflow_store, tmp_path
):
    run_id = _tracked_bundle(
        mlflow_store, tmp_path, state="complete", with_report=True
    )
    mlflow_store.client.log_text(run_id, "not json", "manifest.json")

    with pytest.raises(RunReportError, match=run_id):
        rerender_tracked_run_report(run_id, tracking_store=mlflow_store)

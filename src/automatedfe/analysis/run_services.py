"""Run-ID based analysis recovery and report rerendering services."""

from __future__ import annotations

import tempfile
from os import PathLike
from pathlib import Path

from ..tracking import PROJECT_STATE_TAG, MlflowRunStore
from .run_report import render_run_report


class RunServiceError(RuntimeError):
    """A tracked run could not be operated on safely."""

    def __init__(self, run_id: str, message: str) -> None:
        super().__init__(f"MLflow run {run_id}: {message}")
        self.run_id = run_id


class RunStateError(RunServiceError):
    """A run is not in the state required by an operation."""


class RunAnalysisError(RunServiceError):
    """Analysis retry failed without promoting the run."""


class RunReportError(RunServiceError):
    """Report rerendering failed without replacing the previous report."""


def _resolve_store(
    tracking_store: MlflowRunStore | None,
    *,
    tracking_uri: str | None,
    artifact_root: str | PathLike[str] | None,
) -> MlflowRunStore:
    if tracking_store is not None and (
        tracking_uri is not None or artifact_root is not None
    ):
        raise ValueError(
            "tracking_store cannot be combined with tracking_uri or artifact_root"
        )
    return tracking_store or MlflowRunStore(
        tracking_uri,
        artifact_root=artifact_root,
    )


def _get_run(store: MlflowRunStore, run_id: str):
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    normalized = run_id.strip()
    try:
        return store.get_run(normalized)
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        raise RunServiceError(normalized, f"cannot be read: {error}") from error


def _feature_labels(run, override: str | None) -> str:
    value = (
        override
        if override is not None
        else run.data.params.get("feature_labels", "expression")
    )
    if value not in {"expression", "id"}:
        raise ValueError("feature_labels must be 'expression' or 'id'")
    return value


def analyze_run(
    run_id: str,
    *,
    feature_labels: str | None = None,
    tracking_uri: str | None = None,
    artifact_root: str | PathLike[str] | None = None,
    tracking_store: MlflowRunStore | None = None,
) -> str:
    """Retry analysis for one ``analysis_failed`` run in place.

    Search, materialization, and model evaluation are never invoked. The run
    becomes ``FINISHED/complete`` only after the rendered bundle is validated
    and uploaded successfully. Any repeated failure leaves it
    ``FAILED/analysis_failed``.
    """

    store = _resolve_store(
        tracking_store,
        tracking_uri=tracking_uri,
        artifact_root=artifact_root,
    )
    run = _get_run(store, run_id)
    normalized = run.info.run_id
    state = run.data.tags.get(PROJECT_STATE_TAG)
    if state != "analysis_failed":
        raise RunStateError(
            normalized,
            f"analysis retry requires state 'analysis_failed', found {state!r}",
        )
    labels = _feature_labels(run, feature_labels)

    try:
        with tempfile.TemporaryDirectory(
            prefix=f"automatedfe-analyze-{normalized}-"
        ) as temporary:
            bundle = store.download_artifact_bundle(normalized, Path(temporary))
            if bundle.state != "search_complete":
                raise ValueError(
                    f"completed search artifacts are required, found {bundle.state!r}"
                )
            render_run_report(bundle.path, feature_labels=labels)
            store.log_artifact_bundle(normalized, bundle.path)
    except KeyboardInterrupt:
        store.terminate_run(normalized, "analysis_failed")
        raise
    except BaseException as error:
        store.terminate_run(normalized, "analysis_failed")
        raise RunAnalysisError(normalized, f"analysis retry failed: {error}") from error

    store.terminate_run(normalized, "complete")
    return normalized


def rerender_run_report(
    run_id: str,
    *,
    feature_labels: str | None = None,
    tracking_uri: str | None = None,
    artifact_root: str | PathLike[str] | None = None,
    tracking_store: MlflowRunStore | None = None,
) -> str:
    """Rerender a completed run's report in place from persisted tables."""

    store = _resolve_store(
        tracking_store,
        tracking_uri=tracking_uri,
        artifact_root=artifact_root,
    )
    run = _get_run(store, run_id)
    normalized = run.info.run_id
    state = run.data.tags.get(PROJECT_STATE_TAG)
    if state != "complete" or run.info.status != "FINISHED":
        raise RunStateError(
            normalized,
            "report rerender requires a FINISHED run in state 'complete'; "
            f"found {run.info.status}/{state}",
        )
    labels = _feature_labels(run, feature_labels)

    try:
        with tempfile.TemporaryDirectory(
            prefix=f"automatedfe-report-{normalized}-"
        ) as temporary:
            bundle = store.download_artifact_bundle(normalized, Path(temporary))
            if bundle.state != "search_complete":
                raise ValueError(
                    f"completed search artifacts are required, found {bundle.state!r}"
                )
            render_run_report(bundle.path, feature_labels=labels)
            store.log_artifact_bundle(normalized, bundle.path)
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        raise RunReportError(normalized, f"report rerender failed: {error}") from error

    return normalized


__all__ = [
    "RunAnalysisError",
    "RunReportError",
    "RunServiceError",
    "RunStateError",
    "analyze_run",
    "rerender_run_report",
]

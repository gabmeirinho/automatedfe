"""The canonical MLflow repository for AutomatedFE run data.

The repository deliberately uses :class:`mlflow.tracking.MlflowClient`
instead of MLflow's process-global active-run API.  Different stores can
therefore be used safely in the same process (particularly in tests), and a
tracking URI override never mutates global MLflow configuration.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from numbers import Real
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mlflow.entities import Metric, Param, Run, RunTag, ViewType
from mlflow.tracking import MlflowClient

from ..data.sorting import DEFAULT_RESULTS_DIR

if TYPE_CHECKING:
    from ..analysis.run_bundle import RunBundle

EXPERIMENT_NAME: Final[str] = "automatedfe"
DEFAULT_DATABASE_PATH: Final[Path] = DEFAULT_RESULTS_DIR / "mlflow.db"
DEFAULT_ARTIFACT_ROOT: Final[Path] = DEFAULT_RESULTS_DIR / "artifacts"
DEFAULT_TRACKING_URI: Final[str] = f"sqlite:///{DEFAULT_DATABASE_PATH}"

STRATEGY_TAG: Final[str] = "strategy"
STRATEGY_GROUP_TAG: Final[str] = "strategy_group"
PROJECT_STATE_TAG: Final[str] = "project_state"
FINGERPRINT_TAG_PREFIX: Final[str] = "fingerprint."

_TERMINAL_STATUS: Final[dict[str, str]] = {
    "complete": "FINISHED",
    "search_failed": "FAILED",
    "analysis_failed": "FAILED",
    "interrupted": "KILLED",
}


class TrackingStoreError(RuntimeError):
    """Raised when the tracking backend or artifact repository is unusable."""


def resolve_tracking_uri(tracking_uri: str | None = None) -> str:
    """Resolve an explicit URI, then ``MLFLOW_TRACKING_URI``, then the default."""

    value = (
        tracking_uri if tracking_uri is not None else os.getenv("MLFLOW_TRACKING_URI")
    )
    if value is None:
        return DEFAULT_TRACKING_URI
    if not isinstance(value, str) or not value.strip():
        raise ValueError("tracking_uri must be a non-empty string")
    return value.strip()


def build_run_name(
    strategy: str,
    seed: int,
    *,
    timestamp: datetime | None = None,
) -> str:
    """Build ``{UTC timestamp}_{strategy}_seed{seed}`` for display in MLflow."""

    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError("strategy must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    current = timestamp or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    utc = current.astimezone(UTC)
    # Microseconds make names useful to humans during rapid local runs. MLflow's
    # immutable run ID remains the actual collision-proof identity.
    text = utc.strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{text}_{strategy.strip()}_seed{seed}"


def _metadata_value(value: object) -> str:
    if value is None or isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _fingerprint_value(value: object) -> str:
    if isinstance(value, Mapping):
        value = value.get("fingerprint")
    if not isinstance(value, str) or not value:
        raise ValueError(
            "fingerprints must be non-empty strings or fingerprint records"
        )
    return value


class MlflowRunStore:
    """Create, query, and persist runs in one fixed MLflow experiment.

    Construction performs a real write/read/delete probe.  This intentionally
    happens before callers construct datasets or search algorithms, ensuring
    both the backend and its configured artifact store are available.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        *,
        artifact_root: str | PathLike[str] | None = None,
        validate: bool = True,
    ) -> None:
        self.tracking_uri = resolve_tracking_uri(tracking_uri)
        self.experiment_name = EXPERIMENT_NAME
        self.artifact_root = (
            DEFAULT_ARTIFACT_ROOT
            if artifact_root is None
            else Path(artifact_root).expanduser().resolve()
        )
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        try:
            self._prepare_local_storage()
            self.experiment_id = self._ensure_experiment()
            if validate:
                self.validate_storage()
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise TrackingStoreError(
                f"MLflow tracking is unavailable at {self.tracking_uri!r}: {error}"
            ) from error

    @property
    def artifact_location(self) -> str:
        """Return the local artifact root as an absolute file URI."""

        return self.artifact_root.as_uri()

    def _prepare_local_storage(self) -> None:
        # The default and SQLite overrides need their parent before Alembic can
        # initialize the database. This does not pretend the backend works;
        # validate_storage performs an end-to-end probe below.
        if self.tracking_uri.startswith("sqlite:///"):
            database_text = self.tracking_uri[len("sqlite:///") :].split("?", 1)[0]
            if database_text and database_text != ":memory:":
                Path("/" + database_text.lstrip("/")).expanduser().parent.mkdir(
                    parents=True, exist_ok=True
                )
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        if not self.artifact_root.is_dir():
            raise NotADirectoryError(
                f"MLflow artifact root is not a directory: {self.artifact_root}"
            )

    def _ensure_experiment(self) -> str:
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is not None:
            return experiment.experiment_id
        try:
            return self.client.create_experiment(
                self.experiment_name,
                artifact_location=self.artifact_location,
            )
        except BaseException:
            # A concurrent process may have created the fixed experiment after
            # our read. Re-read before surfacing the original failure.
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                raise
            return experiment.experiment_id

    def validate_storage(self) -> None:
        """Exercise metadata and artifact write/read paths, leaving no active run."""

        probe: Run | None = None
        with tempfile.TemporaryDirectory(prefix="automatedfe-mlflow-probe-") as root:
            source = Path(root) / "probe.txt"
            source.write_text("automatedfe tracking probe\n", encoding="utf-8")
            try:
                probe = self.client.create_run(
                    self.experiment_id,
                    tags={"automatedfe.storage_probe": "true"},
                    run_name="automatedfe-storage-probe",
                )
                run_id = probe.info.run_id
                self.client.log_artifact(run_id, str(source), ".store-health")
                downloaded = Path(
                    self.client.download_artifacts(
                        run_id, ".store-health/probe.txt", str(Path(root) / "download")
                    )
                )
                if downloaded.read_bytes() != source.read_bytes():
                    raise OSError("artifact-store probe returned different content")
            finally:
                if probe is not None:
                    with contextlib.suppress(BaseException):
                        self.client.set_terminated(probe.info.run_id, status="FINISHED")
                    with contextlib.suppress(BaseException):
                        self.client.delete_run(probe.info.run_id)

    def create_run(
        self,
        strategy: str,
        seed: int,
        *,
        parameters: Mapping[str, object] | None = None,
        fingerprints: Mapping[str, object] | None = None,
        strategy_group: str | None = None,
        tags: Mapping[str, object] | None = None,
        timestamp: datetime | None = None,
    ) -> Run:
        """Create a RUNNING run and log its immutable metadata."""

        run_tags = {STRATEGY_TAG: strategy}
        if strategy_group is not None:
            if not isinstance(strategy_group, str) or not strategy_group:
                raise ValueError("strategy_group must be a non-empty string")
            run_tags[STRATEGY_GROUP_TAG] = strategy_group
        if fingerprints:
            run_tags.update(
                {
                    f"{FINGERPRINT_TAG_PREFIX}{name}": _fingerprint_value(value)
                    for name, value in fingerprints.items()
                }
            )
        if tags:
            run_tags.update(
                {str(name): _metadata_value(value) for name, value in tags.items()}
            )
        run = self.client.create_run(
            self.experiment_id,
            tags=run_tags,
            run_name=build_run_name(strategy, seed, timestamp=timestamp),
        )
        if parameters:
            try:
                self.log_parameters(run.info.run_id, parameters)
            except BaseException:
                with contextlib.suppress(BaseException):
                    self.client.set_terminated(run.info.run_id, status="FAILED")
                raise
        return self.client.get_run(run.info.run_id)

    def log_parameters(self, run_id: str, parameters: Mapping[str, object]) -> None:
        """Log configuration parameters without relying on an active run."""

        values = [
            Param(str(name), _metadata_value(value))
            for name, value in parameters.items()
        ]
        if values:
            self.client.log_batch(run_id, params=values)

    def log_fingerprints(self, run_id: str, fingerprints: Mapping[str, object]) -> None:
        """Add content fingerprints as searchable run tags."""

        values = [
            RunTag(f"{FINGERPRINT_TAG_PREFIX}{name}", _fingerprint_value(value))
            for name, value in fingerprints.items()
        ]
        if values:
            self.client.log_batch(run_id, tags=values)

    def log_generation_metrics(
        self,
        run_id: str,
        generation: int,
        metrics: Mapping[str, Real],
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Log one generation as MLflow metric history at ``step=generation``."""

        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValueError("generation must be a non-negative integer")
        instant = timestamp or datetime.now(UTC)
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        timestamp_ms = int(instant.timestamp() * 1000)
        values: list[Metric] = []
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"metric {name!r} must be numeric")
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(f"metric {name!r} must be finite")
            values.append(Metric(str(name), converted, timestamp_ms, generation))
        if values:
            self.client.log_batch(run_id, metrics=values)

    def log_final_metrics(
        self,
        run_id: str,
        metrics: Mapping[str, Real],
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Log final evaluation metrics as run-level MLflow metrics."""

        instant = timestamp or datetime.now(UTC)
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        timestamp_ms = int(instant.timestamp() * 1000)
        values: list[Metric] = []
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"metric {name!r} must be numeric")
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(f"metric {name!r} must be finite")
            values.append(Metric(str(name), converted, timestamp_ms, 0))
        if values:
            self.client.log_batch(run_id, metrics=values)

    def set_project_state(self, run_id: str, project_state: str) -> None:
        if not isinstance(project_state, str) or not project_state:
            raise ValueError("project_state must be a non-empty string")
        self.client.set_tag(run_id, PROJECT_STATE_TAG, project_state)

    def terminate_run(self, run_id: str, project_state: str) -> Run:
        """Persist the terminal project state and corresponding MLflow status."""

        try:
            status = _TERMINAL_STATUS[project_state]
        except KeyError as error:
            expected = ", ".join(_TERMINAL_STATUS)
            raise ValueError(
                f"unknown terminal project state {project_state!r}; expected: {expected}"
            ) from error
        self.client.set_tag(run_id, PROJECT_STATE_TAG, project_state)
        self.client.set_terminated(run_id, status=status)
        return self.client.get_run(run_id)

    def get_run(self, run_id: str, *, include_deleted: bool = False) -> Run:
        run = self.client.get_run(run_id)
        if not include_deleted and run.info.lifecycle_stage != "active":
            raise TrackingStoreError(f"MLflow run {run_id!r} is deleted")
        return run

    def search_runs(
        self,
        *,
        project_states: Sequence[str] | None = None,
        strategies: Sequence[str] | None = None,
        strategy_groups: Sequence[str] | None = None,
        include_deleted: bool = False,
        max_results: int = 1000,
    ) -> list[Run]:
        """Return newest runs, excluding deleted runs unless explicitly requested."""

        view_type = ViewType.ALL if include_deleted else ViewType.ACTIVE_ONLY
        runs = list(
            self.client.search_runs(
                [self.experiment_id],
                run_view_type=view_type,
                max_results=max_results,
                order_by=["attributes.start_time DESC"],
            )
        )
        state_set = None if project_states is None else set(project_states)
        strategy_set = None if strategies is None else set(strategies)
        group_set = None if strategy_groups is None else set(strategy_groups)
        return [
            run
            for run in runs
            if (state_set is None or run.data.tags.get(PROJECT_STATE_TAG) in state_set)
            and (
                strategy_set is None or run.data.tags.get(STRATEGY_TAG) in strategy_set
            )
            and (
                group_set is None or run.data.tags.get(STRATEGY_GROUP_TAG) in group_set
            )
        ]

    def delete_run(self, run_id: str) -> None:
        self.client.delete_run(run_id)

    def log_artifact_bundle(
        self,
        run_id: str,
        staging_directory: str | PathLike[str],
    ) -> None:
        """Validate and upload a bundle, then remove its sole staging copy.

        The source is retained if upload or downloaded-copy validation fails.
        MLflow's generic artifact API is used, so no serialized model or model
        flavor metadata is created.
        """

        from ..analysis.run_bundle import validate_run_bundle

        staging = Path(staging_directory).expanduser().resolve()
        if not staging.is_dir() or staging.is_symlink():
            raise ValueError(f"bundle staging directory does not exist: {staging}")
        source_bundle = validate_run_bundle(staging, validate_dataset=False)
        if source_bundle.run_id != run_id:
            raise ValueError(
                f"bundle run ID {source_bundle.run_id!r} does not match MLflow run ID {run_id!r}"
            )
        self.client.log_artifacts(run_id, str(staging))
        with tempfile.TemporaryDirectory(
            prefix=f"automatedfe-{run_id}-verify-"
        ) as root:
            downloaded = Path(self.client.download_artifacts(run_id, "", root))
            validate_run_bundle(downloaded, validate_dataset=False)
        shutil.rmtree(staging)

    def download_artifact_bundle(
        self,
        run_id: str,
        destination: str | PathLike[str],
    ) -> RunBundle:
        """Download a run's artifacts by immutable ID and validate the bundle."""

        from ..analysis.run_bundle import validate_run_bundle

        self.get_run(run_id)
        target = Path(destination).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        downloaded = Path(self.client.download_artifacts(run_id, "", str(target)))
        try:
            bundle = validate_run_bundle(downloaded, validate_dataset=False)
        except BaseException as error:
            raise TrackingStoreError(
                f"Invalid artifact bundle for MLflow run {run_id!r}: {error}"
            ) from error
        if bundle.run_id != run_id:
            raise TrackingStoreError(
                f"Downloaded artifact bundle run ID {bundle.run_id!r} does not match {run_id!r}"
            )
        return bundle

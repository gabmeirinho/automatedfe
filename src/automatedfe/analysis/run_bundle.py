"""Atomic, schema-aware storage for one feature-search run.

The search code records evidence in memory (and, for candidates, optionally
incrementally on disk).  This module is the boundary between that evidence
and permanent storage.  A bundle is assembled in a uniquely named sibling
directory and is published with one directory rename only after all of its
files and its final manifest validate.

Search failures and interruptions are useful evidence too, but they are not
successful runs.  They are therefore published below ``partial/`` and never
receive a final archive.  The loader validates both the input records and the
files named by the manifest, so a path that merely happens to exist is never
accepted as integrity evidence.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any, Final

from ..data.encoding import DEFAULT_MAPPING_OUTPUT, load_label_mapping
from ..data.transaction_materialization import read_manifest
from ..search.archive import (
    SNAPSHOT_MAPPING_REFERENCE,
    load_archive,
    load_snapshot,
)
from .artifacts import (
    ARCHIVE_SNAPSHOTS_DIRECTORY,
    CANDIDATES_FILENAME,
    DATASET_RECORD_FILENAME,
    FINAL_ARCHIVE_FILENAME,
    GENERATIONS_FILENAME,
    MAPPING_RECORD_FILENAME,
    MMAP_MANIFEST_RECORD_FILENAME,
    RUN_MANIFEST_FILENAME,
    RUN_STATES,
    RUN_STATUS_FILENAME,
    archive_snapshot_path,
    build_dataset_record,
    build_mapping_record,
    build_mmap_manifest_record,
    build_run_manifest,
    build_status,
    canonical_json_text,
    fingerprint_file,
    fingerprint_mapping,
    fingerprint_mmap_manifest,
    load_run_manifest,
    load_status,
    read_candidates_csv,
    read_generations_csv,
    write_candidates_csv,
    write_generations_csv,
    write_run_manifest,
    write_status,
)
from .run_tables import (
    CORRELATIONS_FILENAME,
    FEATURES_FILENAME,
    FinalEvaluationTables,
    IMPORTANCES_FILENAME,
    METRICS_FILENAME,
    TIMINGS_FILENAME,
    read_final_evaluation_tables,
    write_final_evaluation_tables,
)

PARTIAL_DIRECTORY: Final[str] = "partial"
STAGING_MARKER_FILENAME: Final[str] = ".run-bundle-staging"
RUN_BUNDLE_FORMAT: Final[str] = "automatedfe-run-bundle"
RUN_BUNDLE_SCHEMA_VERSION: Final[int] = 1


class RunBundleError(ValueError):
    """Raised when a run bundle cannot be assembled or validated."""


class RunBundleValidationError(RunBundleError):
    """Raised when a named bundle artifact is absent, corrupt, or tampered."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _as_resolved_path(path: str | PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def _json_document(path: Path, document: object) -> Path:
    """Write JSON atomically inside a staging directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            json.dump(document, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary_name)
        raise
    return path


def _read_json(path: Path, artifact_name: str) -> object:
    try:
        with path.open(encoding="utf-8") as source:
            return json.load(source)
    except FileNotFoundError as error:
        raise RunBundleValidationError(
            f"Missing bundle artifact {artifact_name!r}: {path}"
        ) from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunBundleValidationError(
            f"Corrupt bundle artifact {artifact_name!r}: {path}: {error}"
        ) from error


def _relative_artifact(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise RunBundleError(
            f"Bundle artifact is outside its staging directory: {path}"
        ) from error
    if str(relative) in ("", "."):
        raise RunBundleError("Bundle artifact path cannot be the bundle directory")
    return relative.as_posix()


def _safe_manifest_relative_path(root: Path, value: object, artifact_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RunBundleValidationError(
            f"Bundle artifact {artifact_name!r} must name a relative file"
        )
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise RunBundleValidationError(
            f"Bundle artifact {artifact_name!r} escapes the bundle directory: {value!r}"
        ) from error
    if candidate == root.resolve() or not candidate.is_file():
        raise RunBundleValidationError(
            f"Missing bundle artifact {artifact_name!r}: {candidate}"
        )
    return candidate


def _copy_json_record(path: Path, record: Mapping[str, object]) -> None:
    _json_document(path, dict(record))


def _resolve_mapping_argument(
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None,
) -> Mapping[str, Mapping[str, int]]:
    if mapping is None:
        return load_label_mapping(DEFAULT_MAPPING_OUTPUT)
    if isinstance(mapping, (str, PathLike)):
        return load_label_mapping(_as_resolved_path(mapping))
    if not isinstance(mapping, Mapping):
        raise TypeError(
            f"mapping must be a JSON object or a path, got {type(mapping).__name__}"
        )
    return mapping


def _exception_text(error: BaseException | str | None) -> str:
    if error is None:
        return ""
    if isinstance(error, str):
        return error
    message = str(error)
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


@dataclass(frozen=True, slots=True)
class RunBundle:
    """A validated, reopened run bundle."""

    path: Path
    manifest: dict[str, object]
    status: dict[str, object]
    candidates: tuple[dict[str, str], ...]
    generations: tuple[dict[str, str], ...]
    snapshots: tuple[dict[str, object], ...]
    final_archive: dict[str, object] | None
    evaluation: FinalEvaluationTables | None

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @property
    def strategy(self) -> str:
        return str(self.manifest["strategy"])

    @property
    def state(self) -> str:
        return str(self.status["state"])

    @property
    def is_partial(self) -> bool:
        return self.path.parent.name == PARTIAL_DIRECTORY

    @property
    def published_path(self) -> Path:
        """Return the permanent path under which this bundle was published."""

        return self.path

    @property
    def mapping(self) -> Mapping[str, Mapping[str, int]]:
        inputs = self.manifest["inputs"]
        return inputs["mapping"]["mapping"]  # type: ignore[return-value,index]


class RunBundleWriter:
    """Assemble a run in a private staging directory and publish it safely."""

    def __init__(
        self,
        destination: str | PathLike[str],
        *,
        run_id: str | None = None,
        strategy: str,
        dataset_path: str | PathLike[str] | None = None,
        mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
        mmap_dir: str | PathLike[str] | None = None,
        inputs: Mapping[str, Mapping[str, object]] | None = None,
        configuration: Mapping[str, object] | None = None,
        created_at_utc: str | None = None,
        force: bool = False,
        partial_directory: str = PARTIAL_DIRECTORY,
    ) -> None:
        if not isinstance(strategy, str) or not strategy:
            raise ValueError("strategy must be a non-empty string")
        if not isinstance(force, bool):
            raise ValueError("force must be a boolean")
        if not isinstance(partial_directory, str) or not partial_directory:
            raise ValueError("partial_directory must be a non-empty string")
        if Path(partial_directory).name != partial_directory or partial_directory in {
            ".",
            "..",
        }:
            raise ValueError("partial_directory must be a single safe directory name")
        self.destination = _as_resolved_path(destination)
        self.run_id = run_id or self.destination.name
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be a non-empty string")
        self.strategy = strategy
        self.configuration = (
            dict(configuration) if configuration is not None else None
        )
        self.created_at_utc = created_at_utc or _utc_now()
        self.force = force
        self.partial_directory = partial_directory
        self.lifecycle: Any | None = None
        self._evaluation_artifacts: dict[str, str] | None = None
        self._published_path: Path | None = None
        self._closed = False

        if self.destination.exists() and not force:
            raise FileExistsError(
                f"Refusing to overwrite existing run bundle without force=True: {self.destination}"
            )
        if self.destination.exists() and not self.destination.is_dir():
            raise ValueError(
                f"Bundle destination must be a directory: {self.destination}"
            )
        if self.destination.is_symlink():
            raise ValueError(
                f"Bundle destination must not be a symbolic link: {self.destination}"
            )

        self._staging_parent = self.destination.parent
        self._staging_parent.mkdir(parents=True, exist_ok=True)
        staging_name = f".{self.destination.name}.{uuid.uuid4().hex}.staging"
        self.staging_dir = self._staging_parent / staging_name
        self.staging_dir.mkdir()
        self._staging_marker = self.staging_dir / STAGING_MARKER_FILENAME
        self._staging_marker.write_text(staging_name, encoding="utf-8")

        try:
            if inputs is None:
                if dataset_path is None:
                    raise ValueError(
                        "dataset_path is required when inputs are not supplied"
                    )
                if mmap_dir is None:
                    raise ValueError(
                        "mmap_dir is required when inputs are not supplied"
                    )
                resolved_mapping = _resolve_mapping_argument(mapping)
                inputs = {
                    "dataset": build_dataset_record(dataset_path),
                    "mapping": build_mapping_record(resolved_mapping),
                    "mmap_manifest": build_mmap_manifest_record(mmap_dir),
                }
            self.inputs = self._validate_input_documents(inputs)
            self.staged_candidates_path = self.staging_dir / CANDIDATES_FILENAME
            self.staged_generations_path = self.staging_dir / GENERATIONS_FILENAME
            # A running status is useful while the writer is alive.  It is not
            # published until finalize() writes the manifest and renames the
            # complete staging directory.
            write_status(
                self.staging_dir,
                build_status(
                    run_id=self.run_id, state="running", updated_at_utc=_utc_now()
                ),
            )
            self._write_input_records()
        except BaseException:
            self.cleanup()
            raise

    @staticmethod
    def _validate_input_documents(
        inputs: Mapping[str, Mapping[str, object]],
    ) -> dict[str, dict[str, object]]:
        required = {"dataset", "mapping", "mmap_manifest"}
        if set(inputs) != required:
            raise ValueError(
                "Bundle inputs must contain exactly: dataset, mapping, mmap_manifest"
            )
        result = {name: dict(record) for name, record in inputs.items()}
        # Recompute content fingerprints before any search work starts.
        mapping_record = result["mapping"]
        mmap_record = result["mmap_manifest"]
        mapping_content = mapping_record.get("mapping")
        if not isinstance(mapping_content, Mapping) or mapping_record.get(
            "fingerprint"
        ) != fingerprint_mapping(mapping_content):
            raise ValueError("Bundle input record 'mapping' fingerprint mismatch")
        mmap_content = mmap_record.get("manifest")
        if not isinstance(mmap_content, Mapping) or mmap_record.get(
            "fingerprint"
        ) != fingerprint_mmap_manifest(mmap_content):
            raise ValueError("Bundle input record 'mmap_manifest' fingerprint mismatch")
        dataset = result["dataset"]
        if dataset.get("fingerprint") is None or dataset.get("bytes") is None:
            raise ValueError(
                "Bundle input record 'dataset' is missing fingerprint metadata"
            )
        return result

    def _write_input_records(self) -> None:
        _copy_json_record(
            self.staging_dir / DATASET_RECORD_FILENAME, self.inputs["dataset"]
        )
        # The manifest is the authoritative run-level mapping.  The input
        # record is still emitted for a portable, named input artifact and is
        # checked against that same record when the bundle is reopened.
        _copy_json_record(
            self.staging_dir / MAPPING_RECORD_FILENAME, self.inputs["mapping"]
        )
        _copy_json_record(
            self.staging_dir / MMAP_MANIFEST_RECORD_FILENAME,
            self.inputs["mmap_manifest"],
        )

    def write_candidates(self, rows: Sequence[Mapping[str, object]]) -> Path:
        self._ensure_open()
        return write_candidates_csv(self.staged_candidates_path, rows)

    def write_generations(self, rows: Sequence[Mapping[str, object]]) -> Path:
        self._ensure_open()
        return write_generations_csv(self.staged_generations_path, rows)

    def write_snapshot(self, generation: int, document: Mapping[str, object]) -> Path:
        self._ensure_open()
        path = archive_snapshot_path(self.staging_dir, generation)
        return _json_document(path, dict(document))

    def write_final_archive(
        self, document: Mapping[str, object] | str | PathLike[str]
    ) -> Path:
        self._ensure_open()
        path = self.staging_dir / FINAL_ARCHIVE_FILENAME
        if isinstance(document, (str, PathLike)):
            source = _as_resolved_path(document)
            if not source.is_file():
                raise FileNotFoundError(f"Final archive does not exist: {source}")
            try:
                value = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RunBundleError(
                    f"Final archive is not valid JSON: {source}: {error}"
                ) from error
        else:
            value = dict(document)
        if not isinstance(value, Mapping):
            raise TypeError("final archive document must be a JSON object")
        return _json_document(path, value)

    def write_evaluation_tables(self, tables: FinalEvaluationTables) -> dict[str, str]:
        """Persist model-free final-evaluation tables in staging."""

        self._ensure_open()
        if not isinstance(tables, FinalEvaluationTables):
            raise TypeError("tables must be a FinalEvaluationTables instance")
        self._evaluation_artifacts = write_final_evaluation_tables(
            self.staging_dir,
            tables,
        )
        return dict(self._evaluation_artifacts)

    def write_evaluation(
        self,
        evaluation: object,
        *,
        search_fold_metric: str | None = None,
    ) -> dict[str, str]:
        """Persist diagnostics from a final evaluation without model state."""

        from .run_tables import build_final_evaluation_tables

        return self.write_evaluation_tables(
            build_final_evaluation_tables(
                evaluation,
                search_fold_metric=search_fold_metric,
            )
        )

    def write_lifecycle(
        self,
        lifecycle: Any,
        *,
        final_archive: Mapping[str, object] | str | PathLike[str] | None = None,
        include_final_archive: bool = True,
    ) -> None:
        """Persist all evidence currently held by a lifecycle recorder."""

        self._ensure_open()
        if lifecycle is None:
            raise TypeError("lifecycle must not be None")
        self.write_candidates(getattr(lifecycle, "candidate_rows", ()))
        self.write_generations(getattr(lifecycle, "generation_rows", ()))
        for generation, document in getattr(lifecycle, "snapshot_documents", ()):
            self.write_snapshot(generation, document)
        if include_final_archive:
            if final_archive is not None:
                self.write_final_archive(final_archive)
            else:
                snapshot_documents = getattr(lifecycle, "snapshot_documents", ())
                if snapshot_documents:
                    self.write_final_archive(snapshot_documents[-1][1])

    def finalize(
        self,
        state: str,
        *,
        lifecycle: Any | None = None,
        final_archive: Mapping[str, object] | str | PathLike[str] | None = None,
        error: BaseException | str | None = None,
    ) -> RunBundle:
        """Validate and atomically publish the staged bundle.

        ``search_failed`` and ``interrupted`` are moved below ``partial/``.
        They retain candidates, generations, and snapshots but intentionally
        omit ``archive/final.json`` because no successful final result exists.
        """

        self._ensure_open()
        if state not in RUN_STATES:
            raise ValueError(
                f"Unknown run bundle state {state!r}; expected one of: {', '.join(RUN_STATES)}"
            )
        try:
            final_path = self.staging_dir / FINAL_ARCHIVE_FILENAME
            if state != "search_complete" and final_path.exists():
                final_path.unlink()
            if state != "search_complete":
                evaluation_directory = self.staging_dir / "evaluation"
                if evaluation_directory.exists():
                    shutil.rmtree(evaluation_directory)
                self._evaluation_artifacts = None
            if lifecycle is not None:
                self.write_lifecycle(
                    lifecycle,
                    final_archive=final_archive,
                    include_final_archive=state == "search_complete",
                )
            elif state == "search_complete" and final_archive is not None:
                self.write_final_archive(final_archive)

            # Always provide readable empty tables for a setup failure that
            # happened before lifecycle construction.
            if not self.staged_candidates_path.is_file():
                self.write_candidates(())
            if not self.staged_generations_path.is_file():
                self.write_generations(())

            status = build_status(
                run_id=self.run_id,
                state=state,
                updated_at_utc=_utc_now(),
                error=_exception_text(error),
            )
            write_status(self.staging_dir, status)
            manifest = self._build_manifest(state)
            write_run_manifest(self.staging_dir, manifest)

            # Validate before the directory becomes visible as a permanent
            # run. This catches bad paths and corrupt JSON with artifact names.
            _load_run_bundle(self.staging_dir, validate_dataset=False)
            publish_path = self._publish_path(state)
            self._publish(publish_path)
            with contextlib.suppress(FileNotFoundError):
                (publish_path / STAGING_MARKER_FILENAME).unlink()
            self._closed = True
            self._published_path = publish_path
            return _load_run_bundle(publish_path, validate_dataset=False)
        except BaseException:
            self.cleanup()
            raise

    def _build_manifest(
        self,
        state: str,
    ) -> dict[str, object]:
        snapshot_paths = (
            sorted(
                _relative_artifact(path, self.staging_dir)
                for path in (self.staging_dir / ARCHIVE_SNAPSHOTS_DIRECTORY).glob(
                    "generation_*.json"
                )
                if path.is_file()
            )
            if (self.staging_dir / ARCHIVE_SNAPSHOTS_DIRECTORY).is_dir()
            else []
        )
        final_path = self.staging_dir / FINAL_ARCHIVE_FILENAME
        artifacts: dict[str, object] = {
            "candidates": _relative_artifact(
                self.staged_candidates_path, self.staging_dir
            ),
            "generations": _relative_artifact(
                self.staged_generations_path, self.staging_dir
            ),
            "final_archive": (
                _relative_artifact(final_path, self.staging_dir)
                if final_path.is_file()
                else None
            ),
            "archive_snapshots": snapshot_paths,
            "status": _relative_artifact(
                self.staging_dir / RUN_STATUS_FILENAME, self.staging_dir
            ),
            "evaluation": (
                dict(self._evaluation_artifacts)
                if self._evaluation_artifacts is not None
                else None
            ),
        }
        artifact_fingerprints: dict[str, dict[str, object]] = {}
        for path in self._artifact_files(artifacts):
            relative = _relative_artifact(path, self.staging_dir)
            artifact_fingerprints[relative] = {
                "fingerprint": fingerprint_file(path),
                "bytes": path.stat().st_size,
            }
        manifest = build_run_manifest(
            run_id=self.run_id,
            strategy=self.strategy,
            created_at_utc=self.created_at_utc,
            inputs=self.inputs,
            artifacts=artifacts,
            configuration=self.configuration,
        )
        manifest["state"] = state
        manifest["bundle_format"] = RUN_BUNDLE_FORMAT
        manifest["bundle_schema_version"] = RUN_BUNDLE_SCHEMA_VERSION
        manifest["artifact_fingerprints"] = artifact_fingerprints
        return manifest

    def _artifact_files(self, artifacts: Mapping[str, object]) -> tuple[Path, ...]:
        paths: list[Path] = [
            self.staged_candidates_path,
            self.staged_generations_path,
            self.staging_dir / RUN_STATUS_FILENAME,
            self.staging_dir / DATASET_RECORD_FILENAME,
            self.staging_dir / MAPPING_RECORD_FILENAME,
            self.staging_dir / MMAP_MANIFEST_RECORD_FILENAME,
        ]
        final_archive = artifacts.get("final_archive")
        if isinstance(final_archive, str):
            paths.append(self.staging_dir / final_archive)
        snapshots = artifacts.get("archive_snapshots", ())
        if isinstance(snapshots, list):
            paths.extend(
                self.staging_dir / value
                for value in snapshots
                if isinstance(value, str)
            )
        evaluation = artifacts.get("evaluation")
        if isinstance(evaluation, Mapping):
            paths.extend(
                self.staging_dir / value
                for value in evaluation.values()
                if isinstance(value, str)
            )
        return tuple(paths)

    def _publish_path(self, state: str) -> Path:
        if state not in {"search_failed", "interrupted"}:
            return self.destination
        if self.destination.parent.name == self.partial_directory:
            return self.destination
        return self.destination.parent / self.partial_directory / self.destination.name

    def _publish(self, publish_path: Path) -> None:
        publish_path.parent.mkdir(parents=True, exist_ok=True)
        if publish_path.exists():
            if not self.force:
                raise FileExistsError(
                    f"Refusing to overwrite existing run bundle without force=True: {publish_path}"
                )
            if not publish_path.is_dir():
                raise ValueError(
                    f"Bundle destination must be a directory: {publish_path}"
                )
            if publish_path.is_symlink():
                raise ValueError(
                    f"Bundle destination must not be a symbolic link: {publish_path}"
                )
            backup = (
                publish_path.parent
                / f".{publish_path.name}.{uuid.uuid4().hex}.replaced"
            )
            os.replace(publish_path, backup)
            try:
                os.replace(self.staging_dir, publish_path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.replace(backup, publish_path)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(self.staging_dir, publish_path)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RunBundleError("Run bundle writer is already finalized")
        if not self.staging_dir.is_dir():
            raise RunBundleError(
                f"Run bundle staging directory is unavailable: {self.staging_dir}"
            )

    def cleanup(self) -> None:
        """Remove only this writer's exact, marker-bearing staging directory."""

        if self._closed:
            return
        staging = self.staging_dir
        try:
            resolved = staging.resolve()
            expected_parent = self._staging_parent.resolve()
            marker = resolved / STAGING_MARKER_FILENAME
            safe_name = (
                resolved.parent == expected_parent
                and resolved.name.startswith(f".{self.destination.name}.")
                and resolved.name.endswith(".staging")
                and marker.is_file()
                and marker.read_text(encoding="utf-8") == resolved.name
                and not staging.is_symlink()
            )
            if safe_name:
                shutil.rmtree(resolved)
        except (OSError, UnicodeDecodeError):
            # Cleanup is best effort, but never broadens its target if a
            # staging path has been replaced or malformed.
            return


def _validate_artifact_fingerprints(
    root: Path,
    manifest: Mapping[str, object],
) -> None:
    records = manifest.get("artifact_fingerprints")
    if records is None:
        return
    if not isinstance(records, Mapping):
        raise RunBundleValidationError("Artifact fingerprint metadata is malformed")
    for relative, record in records.items():
        if not isinstance(relative, str) or not isinstance(record, Mapping):
            raise RunBundleValidationError("Artifact fingerprint metadata is malformed")
        path = _safe_manifest_relative_path(root, relative, relative)
        expected = record.get("fingerprint")
        actual = fingerprint_file(path)
        if expected != actual:
            raise RunBundleValidationError(
                f"Artifact fingerprint mismatch for {relative!r}: recorded {expected!r}, computed {actual!r}"
            )
        expected_bytes = record.get("bytes")
        if expected_bytes is not None and expected_bytes != path.stat().st_size:
            raise RunBundleValidationError(
                f"Artifact size mismatch for {relative!r}: recorded {expected_bytes!r}, actual {path.stat().st_size}"
            )


def _validate_input_files(root: Path, manifest: Mapping[str, object]) -> None:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise RunBundleValidationError("Run manifest inputs are malformed")
    for name, relative in (
        ("dataset", DATASET_RECORD_FILENAME),
        ("mapping", MAPPING_RECORD_FILENAME),
        ("mmap_manifest", MMAP_MANIFEST_RECORD_FILENAME),
    ):
        path = _safe_manifest_relative_path(root, relative, f"inputs/{name}")
        data = _read_json(path, f"inputs/{name}")
        if not isinstance(data, Mapping):
            raise RunBundleValidationError(
                f"Input artifact {name!r} is not a JSON object"
            )
        if canonical_json_text(data) != canonical_json_text(inputs[name]):
            raise RunBundleValidationError(
                f"Input artifact {name!r} does not match the run manifest record"
            )


def _validate_available_source_inputs(manifest: Mapping[str, object]) -> None:
    """Cross-check available provenance files without trusting paths alone."""

    inputs = manifest["inputs"]
    mapping_record = inputs["mapping"]
    mapping_source = mapping_record.get("source_path")
    if isinstance(mapping_source, str) and mapping_source:
        path = Path(mapping_source)
        if path.is_file():
            try:
                source_mapping = load_label_mapping(path)
            except (OSError, TypeError, ValueError) as error:
                raise RunBundleValidationError(
                    f"Mapping input is not readable: {path}: {error}"
                ) from error
            actual = fingerprint_mapping(source_mapping)
            if actual != mapping_record["fingerprint"]:
                raise RunBundleValidationError(
                    f"Mapping input fingerprint mismatch: {path}"
                )

    mmap_record = inputs["mmap_manifest"]
    mmap_source = mmap_record.get("source_path")
    if isinstance(mmap_source, str) and mmap_source:
        path = Path(mmap_source)
        if path.is_file():
            try:
                source_manifest = read_manifest(path.parent)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise RunBundleValidationError(
                    f"Mmap manifest input is not readable: {path}: {error}"
                ) from error
            actual = fingerprint_mmap_manifest(source_manifest)
            if actual != mmap_record["fingerprint"]:
                raise RunBundleValidationError(
                    f"Mmap manifest input fingerprint mismatch: {path}"
                )


def _load_run_bundle(
    path: str | PathLike[str],
    *,
    validate_dataset: bool | None = None,
) -> RunBundle:
    root = _as_resolved_path(path)
    if not root.is_dir():
        raise RunBundleValidationError(f"Run bundle directory does not exist: {root}")
    try:
        manifest = load_run_manifest(root, validate_dataset=False)
        dataset_source = manifest["inputs"]["dataset"].get("source_path")
        should_validate_dataset = validate_dataset is True or (
            validate_dataset is None
            and isinstance(dataset_source, str)
            and bool(dataset_source)
        )
        if should_validate_dataset:
            manifest = load_run_manifest(root, validate_dataset=True)
    except (OSError, TypeError, ValueError) as error:
        raise RunBundleValidationError(
            f"Invalid run manifest {root / RUN_MANIFEST_FILENAME}: {error}"
        ) from error
    try:
        status = load_status(root)
    except (OSError, TypeError, ValueError) as error:
        raise RunBundleValidationError(
            f"Invalid bundle artifact {RUN_STATUS_FILENAME!r}: {error}"
        ) from error
    if status["run_id"] != manifest["run_id"]:
        raise RunBundleValidationError(
            f"Status artifact run_id {status['run_id']!r} does not match manifest run_id {manifest['run_id']!r}"
        )
    if manifest.get("state") is not None and manifest["state"] != status["state"]:
        raise RunBundleValidationError("Manifest state does not match status artifact")

    _validate_input_files(root, manifest)
    _validate_available_source_inputs(manifest)
    _validate_artifact_fingerprints(root, manifest)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RunBundleValidationError("Run manifest artifacts are malformed")

    for artifact_name, expected_path in (
        ("candidates", CANDIDATES_FILENAME),
        ("generations", GENERATIONS_FILENAME),
        ("status", RUN_STATUS_FILENAME),
    ):
        if artifacts.get(artifact_name) != expected_path:
            raise RunBundleValidationError(
                f"Bundle artifact {artifact_name!r} must be {expected_path!r}"
            )

    candidates_path = _safe_manifest_relative_path(
        root, artifacts.get("candidates"), "candidates"
    )
    generations_path = _safe_manifest_relative_path(
        root, artifacts.get("generations"), "generations"
    )
    try:
        candidates = tuple(read_candidates_csv(candidates_path))
    except (OSError, ValueError) as error:
        raise RunBundleValidationError(
            f"Invalid bundle artifact 'candidates': {error}"
        ) from error
    try:
        generations = tuple(read_generations_csv(generations_path))
    except (OSError, ValueError) as error:
        raise RunBundleValidationError(
            f"Invalid bundle artifact 'generations': {error}"
        ) from error

    snapshot_names = artifacts.get("archive_snapshots")
    if not isinstance(snapshot_names, list):
        raise RunBundleValidationError(
            "Bundle artifact 'archive_snapshots' must be a list"
        )
    mapping = manifest["inputs"]["mapping"]["mapping"]  # type: ignore[index]
    snapshots: list[dict[str, object]] = []
    snapshot_generations: list[int] = []
    for index, relative in enumerate(snapshot_names):
        snapshot_path = _safe_manifest_relative_path(
            root, relative, f"archive_snapshots[{index}]"
        )
        document = _read_json(snapshot_path, f"archive_snapshots[{index}]")
        if not isinstance(document, dict):
            raise RunBundleValidationError(
                f"Invalid bundle artifact 'archive_snapshots[{index}]': not an object"
            )
        if document.get("mapping_ref") != SNAPSHOT_MAPPING_REFERENCE:
            raise RunBundleValidationError(
                f"Invalid bundle artifact 'archive_snapshots[{index}]': mapping_ref does not resolve the run manifest"
            )
        try:
            load_snapshot(snapshot_path, mapping)
        except (OSError, TypeError, ValueError) as error:
            raise RunBundleValidationError(
                f"Invalid bundle artifact 'archive_snapshots[{index}]': {error}"
            ) from error
        snapshots.append(document)
        name = snapshot_path.name
        if not (name.startswith("generation_") and name.endswith(".json")):
            raise RunBundleValidationError(
                f"Invalid bundle artifact 'archive_snapshots[{index}]': invalid generation name"
            )
        try:
            snapshot_generations.append(int(name[len("generation_") : -5]))
        except ValueError:
            raise RunBundleValidationError(
                f"Invalid bundle artifact 'archive_snapshots[{index}]': invalid generation name"
            )
    if snapshot_generations != sorted(snapshot_generations):
        raise RunBundleValidationError(
            "Bundle archive snapshots are not in generation order"
        )

    final_archive: dict[str, object] | None = None
    final_name = artifacts.get("final_archive")
    if final_name is not None:
        final_path = _safe_manifest_relative_path(root, final_name, "final_archive")
        document = _read_json(final_path, "final_archive")
        if not isinstance(document, dict):
            raise RunBundleValidationError(
                "Invalid bundle artifact 'final_archive': not an object"
            )
        try:
            if document.get("format") == "automatedfe-archive-snapshot":
                if document.get("mapping_ref") != SNAPSHOT_MAPPING_REFERENCE:
                    raise RunBundleValidationError(
                        "final_archive mapping_ref does not resolve the run manifest"
                    )
                load_snapshot(final_path, mapping)
            else:
                load_archive(final_path, mapping=mapping)
        except (OSError, TypeError, ValueError) as error:
            raise RunBundleValidationError(
                f"Invalid bundle artifact 'final_archive': {error}"
            ) from error
        final_archive = document
        if snapshots and document.get("format") == "automatedfe-archive-snapshot":
            if canonical_json_text(document) != canonical_json_text(snapshots[-1]):
                raise RunBundleValidationError(
                    "Bundle artifact 'final_archive' does not match the final archive snapshot"
                )
    elif (
        status["state"] == "search_complete"
        and manifest.get("strategy") != "enumerative_without_archive"
    ):
        # Evaluation-free complete searches have no final archive by design;
        # every evaluated strategy must publish one.
        raise RunBundleValidationError(
            "Missing bundle artifact 'final_archive' for a completed evaluated run"
        )

    if (
        status["state"] in {"search_failed", "interrupted"}
        and final_archive is not None
    ):
        raise RunBundleValidationError(
            f"Partial bundle must not contain a final archive: {final_name}"
        )
    evaluation: FinalEvaluationTables | None = None
    evaluation_artifacts = artifacts.get("evaluation")
    if evaluation_artifacts is not None:
        if not isinstance(evaluation_artifacts, Mapping):
            raise RunBundleValidationError(
                "Bundle artifact 'evaluation' must be an object or null"
            )
        for name, relative in (
            ("features", FEATURES_FILENAME),
            ("metrics", METRICS_FILENAME),
            ("importances", IMPORTANCES_FILENAME),
            ("correlations", CORRELATIONS_FILENAME),
            ("timings", TIMINGS_FILENAME),
        ):
            try:
                _safe_manifest_relative_path(
                    root,
                    evaluation_artifacts.get(name),
                    f"evaluation.{name}",
                )
            except RunBundleValidationError as error:
                raise RunBundleValidationError(str(error)) from error
            if evaluation_artifacts.get(name) != relative:
                raise RunBundleValidationError(
                    f"Bundle artifact evaluation.{name!s} must be {relative!r}"
                )
        try:
            evaluation = read_final_evaluation_tables(
                root,
                evaluation_artifacts,
            )
        except (OSError, ValueError) as error:
            raise RunBundleValidationError(
                f"Invalid bundle artifact 'evaluation': {error}"
            ) from error
        if status["state"] in {"search_failed", "interrupted"}:
            raise RunBundleValidationError(
                "Partial bundle must not contain final evaluation tables"
            )
    if (
        status["state"] in {"search_failed", "interrupted"}
        and (root / "report.html").exists()
    ):
        raise RunBundleValidationError("Partial bundle must not contain 'report.html'")
    return RunBundle(
        path=root,
        manifest=dict(manifest),
        status=dict(status),
        candidates=candidates,
        generations=generations,
        snapshots=tuple(snapshots),
        final_archive=final_archive,
        evaluation=evaluation,
    )


def load_run_bundle(
    path: str | PathLike[str],
    *,
    validate_dataset: bool | None = None,
) -> RunBundle:
    """Open and fully validate a permanent or partial run bundle."""

    return _load_run_bundle(path, validate_dataset=validate_dataset)


def open_run_bundle(
    path: str | PathLike[str],
    *,
    validate_dataset: bool | None = None,
) -> RunBundle:
    """Compatibility alias for :func:`load_run_bundle`."""

    return load_run_bundle(path, validate_dataset=validate_dataset)


def validate_run_bundle(
    path: str | PathLike[str],
    *,
    validate_dataset: bool | None = None,
) -> RunBundle:
    """Validate and return a reopened bundle."""

    return load_run_bundle(path, validate_dataset=validate_dataset)


def create_run_bundle(
    destination: str | PathLike[str],
    **kwargs: object,
) -> RunBundleWriter:
    """Create a staging writer for a run bundle."""

    return RunBundleWriter(destination, **kwargs)  # type: ignore[arg-type]


def finalize_run_bundle(
    writer: RunBundleWriter,
    state: str,
    **kwargs: object,
) -> RunBundle:
    """Finalize a writer created by :func:`create_run_bundle`."""

    if not isinstance(writer, RunBundleWriter):
        raise TypeError("writer must be a RunBundleWriter")
    return writer.finalize(state, **kwargs)  # type: ignore[arg-type]


def write_run_bundle(
    destination: str | PathLike[str],
    *,
    strategy: str,
    dataset_path: str | PathLike[str] | None = None,
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    mmap_dir: str | PathLike[str] | None = None,
    inputs: Mapping[str, Mapping[str, object]] | None = None,
    configuration: Mapping[str, object] | None = None,
    run_id: str | None = None,
    lifecycle: Any | None = None,
    state: str = "search_complete",
    final_archive: Mapping[str, object] | str | PathLike[str] | None = None,
    error: BaseException | str | None = None,
    force: bool = False,
) -> RunBundle:
    """Assemble and publish one bundle in a single call."""

    writer = RunBundleWriter(
        destination,
        strategy=strategy,
        dataset_path=dataset_path,
        mapping=mapping,
        mmap_dir=mmap_dir,
        inputs=inputs,
        configuration=configuration,
        run_id=run_id,
        force=force,
    )
    return writer.finalize(
        state,
        lifecycle=lifecycle,
        final_archive=final_archive,
        error=error,
    )


__all__ = [
    "PARTIAL_DIRECTORY",
    "RUN_BUNDLE_FORMAT",
    "RUN_BUNDLE_SCHEMA_VERSION",
    "RunBundle",
    "RunBundleError",
    "RunBundleValidationError",
    "RunBundleWriter",
    "create_run_bundle",
    "finalize_run_bundle",
    "load_run_bundle",
    "open_run_bundle",
    "validate_run_bundle",
    "write_run_bundle",
]

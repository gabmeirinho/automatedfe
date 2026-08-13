"""Schema-versioned run manifest, input fingerprints, feature IDs, and schemas.

A structured run is a directory owning exactly one schema-versioned
``manifest.json`` (with a sibling ``manifest.json.sha256`` checksum) that
records content fingerprints for the dataset, mapping, and mmap manifest
inputs. The label mapping is stored once at run level inside the manifest;
archive snapshots and the final archive reference it instead of embedding
copies. Integrity evidence is always content-derived: paths are recorded for
provenance only and are never accepted as evidence on their own.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Final

RUN_MANIFEST_FORMAT: Final[str] = "automatedfe-run"
RUN_MANIFEST_SCHEMA_VERSION: Final[int] = 1

STATUS_FORMAT: Final[str] = "automatedfe-status"
STATUS_SCHEMA_VERSION: Final[int] = 1

RUN_STATES: Final[tuple[str, ...]] = (
    "running",
    "search_failed",
    "interrupted",
    "search_complete",
)

# Run-directory-relative artifact paths.
RUN_MANIFEST_FILENAME: Final[str] = "manifest.json"
RUN_MANIFEST_FINGERPRINT_FILENAME: Final[str] = "manifest.json.sha256"
RUN_STATUS_FILENAME: Final[str] = "status.json"
CANDIDATES_FILENAME: Final[str] = "candidates.csv"
GENERATIONS_FILENAME: Final[str] = "generations.csv"
FINAL_ARCHIVE_FILENAME: Final[str] = "archive/final.json"
ARCHIVE_SNAPSHOTS_DIRECTORY: Final[str] = "snapshots"
INPUTS_DIRECTORY: Final[str] = "inputs"
DATASET_RECORD_FILENAME: Final[str] = "inputs/dataset.json"
MAPPING_RECORD_FILENAME: Final[str] = "inputs/mapping.json"
MMAP_MANIFEST_RECORD_FILENAME: Final[str] = "inputs/mmap_manifest.json"

FINGERPRINT_PREFIX: Final[str] = "sha256:"
_FINGERPRINT_LENGTH: Final[int] = len(FINGERPRINT_PREFIX) + 64
_CHUNK_SIZE: Final[int] = 1024 * 1024

# Candidate-level and generation-level CSV schemas. The candidate schema is
# the same table the runner has always written incrementally; loose CSV
# outputs are not structured runs, but their columns follow this contract.
CANDIDATES_COLUMNS: Final[tuple[str, ...]] = (
    "Strategy",
    "CandidateIndex",
    "Generation",
    "Expression",
    "Dependencies",
    "Split1",
    "Split2",
    "Split3",
    "MaterializationTime",
    "ArchiveMember",
    "Status",
    "Error",
)

GENERATIONS_COLUMNS: Final[tuple[str, ...]] = (
    "Strategy",
    "Generation",
    "Generated",
    "Unique",
    "Duplicate",
    "Invalid",
    "Evaluated",
    "ArchiveSize",
    "Added",
    "Removed",
    "DurationSeconds",
    "CumulativeRuntimeSeconds",
)

_FINGERPRINT_REQUIRED_INPUT_KEYS = ("dataset", "mapping", "mmap_manifest")
_ARTIFACT_REQUIRED_KEYS = (
    "candidates",
    "generations",
    "final_archive",
    "archive_snapshots",
    "status",
)


def fingerprint_bytes(data: bytes) -> str:
    """Return the content fingerprint ``sha256:<hex>`` of *data*."""

    return FINGERPRINT_PREFIX + hashlib.sha256(data).hexdigest()


def fingerprint_text(text: str) -> str:
    """Return the content fingerprint of a UTF-8 text string."""

    return fingerprint_bytes(text.encode("utf-8"))


def fingerprint_file(path: str | PathLike[str]) -> str:
    """Return the content fingerprint of a file, hashed in bounded chunks."""

    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        while chunk := file.read(_CHUNK_SIZE):
            digest.update(chunk)
    return FINGERPRINT_PREFIX + digest.hexdigest()


def canonical_json_text(value: object) -> str:
    """Serialize *value* canonically so identical content hashes identically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def fingerprint_mapping(mapping: Mapping[str, Mapping[str, int]]) -> str:
    """Return the content fingerprint of a label mapping.

    The fingerprint covers the mapping content in canonical JSON form, so
    insertion order never matters and any value change changes the
    fingerprint.
    """

    if not isinstance(mapping, Mapping):
        raise TypeError(f"mapping must be a JSON object, got {type(mapping).__name__}")
    return fingerprint_text(canonical_json_text(mapping))


def fingerprint_mmap_manifest(manifest: Mapping[str, object]) -> str:
    """Return the content fingerprint of a materialized-column manifest."""

    if not isinstance(manifest, Mapping):
        raise TypeError(
            f"mmap manifest must be a JSON object, got {type(manifest).__name__}"
        )
    return fingerprint_text(canonical_json_text(manifest))


def fingerprint_expression(expression: object) -> str:
    """Return the content fingerprint of an expression's canonical structure."""

    from ..search.search import canonical_expression_key

    return fingerprint_text(canonical_expression_key(expression))


def feature_id(expression: object) -> str:
    """Return the stable expression-derived feature ID for an expression.

    Identical expressions always produce the identical ID and any structural
    change produces a different ID. The ID derives from the same canonical
    expression key used for search-time deduplication.
    """

    from ..search.search import canonical_expression_key

    digest = hashlib.sha256(
        canonical_expression_key(expression).encode("utf-8")
    ).hexdigest()
    return f"feat_{digest[:12]}"


def build_dataset_record(path: str | PathLike[str]) -> dict[str, object]:
    """Record the dataset input with a content fingerprint, never a path hash."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset parquet file does not exist: {resolved}")
    return {
        "source_path": str(resolved),
        "fingerprint": fingerprint_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def build_mapping_record(
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str],
) -> dict[str, object]:
    """Record the label mapping with its content fingerprint.

    The full mapping metadata is stored once at run level so snapshots can
    reference it instead of embedding copies. When *mapping* is a path, the
    path is recorded for provenance only; the fingerprint always covers the
    loaded mapping content.
    """

    resolved_path: str | None = None
    if isinstance(mapping, (str, PathLike)):
        from ..data.encoding import load_label_mapping

        resolved_path = str(Path(mapping).resolve())
        mapping = load_label_mapping(Path(mapping))
    if not isinstance(mapping, Mapping):
        raise TypeError(
            f"mapping must be a JSON object or a path, got {type(mapping).__name__}"
        )
    return {
        "source_path": resolved_path,
        "fingerprint": fingerprint_mapping(mapping),
        "mapping": {family: dict(values) for family, values in mapping.items()},
    }


def build_mmap_manifest_record(mmap_dir: str | PathLike[str]) -> dict[str, object]:
    """Record the mmap manifest with its content fingerprint."""

    from ..data.transaction_materialization import MANIFEST_FILENAME, read_manifest

    resolved_dir = Path(mmap_dir).resolve()
    manifest_path = resolved_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Mmap manifest file does not exist: {manifest_path}"
        )
    manifest = read_manifest(resolved_dir)
    return {
        "source_path": str(manifest_path),
        "fingerprint": fingerprint_mmap_manifest(manifest),
        "manifest": dict(manifest),
    }


def archive_snapshot_path(
    run_dir: str | PathLike[str],
    generation: int,
) -> Path:
    """Return the run-relative archive snapshot path for *generation*."""

    if isinstance(generation, bool) or not isinstance(generation, int):
        raise ValueError(f"generation must be a non-negative integer, got {generation!r}")
    if generation < 0:
        raise ValueError(f"generation must be a non-negative integer, got {generation}")
    return Path(run_dir) / ARCHIVE_SNAPSHOTS_DIRECTORY / f"generation_{generation:06d}.json"


def generation_from_snapshot_path(path: str | PathLike[str]) -> int | None:
    """Return the generation encoded in a snapshot filename, or None."""

    name = Path(path).name
    prefix = "generation_"
    if not name.startswith(prefix) or not name.endswith(".json"):
        return None
    digits = name[len(prefix) : -len(".json")]
    if not digits.isdigit():
        return None
    return int(digits)


def build_run_manifest(
    *,
    run_id: str,
    strategy: str,
    created_at_utc: str,
    inputs: Mapping[str, object],
    artifacts: Mapping[str, object],
) -> dict[str, object]:
    """Build a schema-versioned run manifest document."""

    document = {
        "format": RUN_MANIFEST_FORMAT,
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "strategy": strategy,
        "created_at_utc": created_at_utc,
        "inputs": {key: dict(value) for key, value in inputs.items()},
        "artifacts": {key: value for key, value in artifacts.items()},
    }
    return _validate_run_manifest(document)


def _validate_run_manifest(data: object) -> dict[str, object]:
    if not isinstance(data, dict):
        raise TypeError(
            f"Run manifest must be a JSON object, got {type(data).__name__}"
        )
    if data.get("format") != RUN_MANIFEST_FORMAT:
        raise ValueError(
            f"Unknown run manifest format: {data.get('format')!r}"
        )
    version = data.get("schema_version")
    if version != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported run manifest schema version {version!r} "
            f"(expected {RUN_MANIFEST_SCHEMA_VERSION})"
        )
    for name in ("run_id", "strategy", "created_at_utc"):
        if not isinstance(data.get(name), str):
            raise TypeError(f"Run manifest is missing its {name!r} string field")
    inputs = data.get("inputs")
    artifacts = data.get("artifacts")
    if not isinstance(inputs, dict):
        raise TypeError("Run manifest is missing its 'inputs' object")
    if not isinstance(artifacts, dict):
        raise TypeError("Run manifest is missing its 'artifacts' object")
    for name in _FINGERPRINT_REQUIRED_INPUT_KEYS:
        if not isinstance(inputs.get(name), dict):
            raise TypeError(
                f"Run manifest inputs are missing their {name!r} fingerprint record"
            )
    for name in _ARTIFACT_REQUIRED_KEYS:
        if name not in artifacts:
            raise TypeError(
                f"Run manifest artifacts are missing their {name!r} entry"
            )
    return data


def _valid_fingerprint(value: object) -> bool:
    if not isinstance(value, str) or len(value) != _FINGERPRINT_LENGTH:
        return False
    if not value.startswith(FINGERPRINT_PREFIX):
        return False
    return all(
        character in "0123456789abcdef"
        for character in value[len(FINGERPRINT_PREFIX) :]
    )


def _validate_input_records(manifest: Mapping[str, object]) -> None:
    inputs = manifest["inputs"]
    for key in ("mapping", "mmap_manifest"):
        record = inputs[key]
        fingerprint = record.get("fingerprint")
        content = record.get("mapping" if key == "mapping" else "manifest")
        if not _valid_fingerprint(fingerprint):
            raise ValueError(
                f"Input record {key!r} must declare a sha256 fingerprint"
            )
        if not isinstance(content, Mapping):
            raise TypeError(
                f"Input record {key!r} must carry its content for validation"
            )
        recomputed = (
            fingerprint_mapping(content)
            if key == "mapping"
            else fingerprint_mmap_manifest(content)
        )
        if recomputed != fingerprint:
            raise ValueError(
                f"Input record {key!r} fingerprint mismatch (tampering): "
                f"recorded {fingerprint}, content hashes to {recomputed}"
            )
    dataset = inputs["dataset"]
    if not _valid_fingerprint(dataset.get("fingerprint")):
        raise ValueError(
            "Input record 'dataset' must declare a sha256 fingerprint"
        )
    source = dataset.get("source_path")
    if source is not None and not isinstance(source, str):
        raise TypeError("Dataset record 'source_path' must be a string or null")
    if isinstance(dataset.get("bytes"), bool) or not isinstance(
        dataset.get("bytes"), int
    ):
        raise TypeError("Dataset record 'bytes' must be a non-negative integer")
    if dataset["bytes"] < 0:
        raise ValueError("Dataset record 'bytes' must be a non-negative integer")


def _validate_dataset_file(manifest: Mapping[str, object]) -> None:
    record = manifest["inputs"]["dataset"]
    source = record.get("source_path")
    if not isinstance(source, str) or not source:
        raise ValueError(
            "Dataset record carries no source path for fingerprint validation"
        )
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset input file does not exist: {source_path}")
    recomputed = fingerprint_file(source_path)
    if recomputed != record["fingerprint"]:
        raise ValueError(
            f"Dataset fingerprint mismatch (tampering): {source_path}"
        )


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary_name)
        raise
    return path


def write_run_manifest(
    run_dir: str | PathLike[str],
    manifest: Mapping[str, object],
    *,
    force: bool = False,
) -> Path:
    """Atomically persist a validated manifest and its checksum record.

    The sibling ``manifest.json.sha256`` records the fingerprint of the exact
    manifest bytes, so later tampering is detectable without trusting paths.
    """

    if not isinstance(force, bool):
        raise ValueError("force must be a boolean")
    resolved_dir = Path(run_dir).resolve()
    manifest_path = resolved_dir / RUN_MANIFEST_FILENAME
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing run manifest without force=True: "
            f"{manifest_path}"
        )
    text = canonical_json_text(_validate_run_manifest(manifest)) + "\n"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(manifest_path, text)
    _atomic_write_text(
        resolved_dir / RUN_MANIFEST_FINGERPRINT_FILENAME,
        fingerprint_text(text) + "\n",
    )
    return manifest_path


def load_run_manifest(
    run_dir: str | PathLike[str],
    *,
    validate_dataset: bool = False,
) -> dict[str, object]:
    """Load and validate the run manifest of a structured run directory.

    Rejects missing or malformed manifests, unsupported schema versions,
    checksum tampering, and inconsistent input fingerprint records. Paths are
    never accepted as integrity evidence: the mapping and mmap manifest
    records are always re-validated against their embedded content, and the
    dataset file fingerprint is re-validated when *validate_dataset* is set.
    """

    resolved_dir = Path(run_dir).resolve()
    manifest_path = resolved_dir / RUN_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(f"Not a structured run: no run manifest at {manifest_path}")
    try:
        raw = manifest_path.read_bytes()
    except OSError as error:
        raise ValueError(f"Cannot read run manifest: {manifest_path}: {error}") from error
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Run manifest is not valid JSON: {manifest_path}: {error}") from error

    validated = _validate_run_manifest(data)

    checksum_path = resolved_dir / RUN_MANIFEST_FINGERPRINT_FILENAME
    if not checksum_path.is_file():
        raise ValueError(
            f"Run manifest is missing its checksum record: {checksum_path}"
        )
    recorded = checksum_path.read_text(encoding="utf-8").strip()
    if recorded != fingerprint_bytes(raw):
        raise ValueError(
            f"Run manifest fingerprint mismatch (tampering): {manifest_path}"
        )

    _validate_input_records(validated)
    if validate_dataset:
        _validate_dataset_file(validated)
    return validated


def build_status(
    *,
    run_id: str,
    state: str,
    updated_at_utc: str | None = None,
    error: str = "",
) -> dict[str, object]:
    """Build a schema-versioned run status document."""

    document = {
        "format": STATUS_FORMAT,
        "schema_version": STATUS_SCHEMA_VERSION,
        "run_id": run_id,
        "state": state,
        "error": error,
        "updated_at_utc": updated_at_utc,
    }
    return _validate_status(document)


def _validate_status(data: object) -> dict[str, object]:
    if not isinstance(data, dict):
        raise TypeError(f"Run status must be a JSON object, got {type(data).__name__}")
    if data.get("format") != STATUS_FORMAT:
        raise ValueError(f"Unknown run status format: {data.get('format')!r}")
    version = data.get("schema_version")
    if version != STATUS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported run status schema version {version!r} "
            f"(expected {STATUS_SCHEMA_VERSION})"
        )
    if not isinstance(data.get("run_id"), str):
        raise TypeError("Run status is missing its 'run_id' string field")
    state = data.get("state")
    if state not in RUN_STATES:
        raise ValueError(
            f"Unknown run status state {state!r}; expected one of: "
            + ", ".join(RUN_STATES)
        )
    if not isinstance(data.get("error"), str):
        raise TypeError("Run status 'error' must be a string")
    updated = data.get("updated_at_utc")
    if updated is not None and not isinstance(updated, str):
        raise TypeError("Run status 'updated_at_utc' must be a string or null")
    return data


def write_status(
    run_dir: str | PathLike[str],
    status: Mapping[str, object],
) -> Path:
    """Atomically persist a validated run status document.

    Status is expected to be updated as a run progresses, so an existing
    status file is replaced.
    """

    resolved_dir = Path(run_dir).resolve()
    status_path = resolved_dir / RUN_STATUS_FILENAME
    text = canonical_json_text(_validate_status(status)) + "\n"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    return _atomic_write_text(status_path, text)


def load_status(run_dir: str | PathLike[str]) -> dict[str, object]:
    """Load and validate the status document of a structured run directory."""

    resolved_dir = Path(run_dir).resolve()
    status_path = resolved_dir / RUN_STATUS_FILENAME
    if not status_path.is_file():
        raise FileNotFoundError(f"Run status file does not exist: {status_path}")
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Run status is not valid JSON: {status_path}: {error}") from error
    return _validate_status(data)


def _write_schema_csv(
    path: str | PathLike[str],
    rows: Sequence[Mapping[str, object]],
    columns: tuple[str, ...],
) -> Path:
    resolved = Path(path).resolve()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != set(columns):
            raise ValueError(
                f"CSV rows must contain exactly the schema columns: {', '.join(columns)}"
            )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns))
    writer.writeheader()
    writer.writerows(rows)
    return _atomic_write_text(resolved, buffer.getvalue())


def _read_schema_csv(
    path: str | PathLike[str],
    columns: tuple[str, ...],
) -> list[dict[str, str]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {resolved}")
    with open(resolved, encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(
                f"CSV header must be the schema columns: {', '.join(columns)}"
            )
        return [dict(row) for row in reader]


def write_candidates_csv(
    path: str | PathLike[str],
    rows: Sequence[Mapping[str, object]],
) -> Path:
    """Atomically write candidate rows following the candidates schema."""

    return _write_schema_csv(path, rows, CANDIDATES_COLUMNS)


def read_candidates_csv(path: str | PathLike[str]) -> list[dict[str, str]]:
    """Read and schema-validate a candidates CSV."""

    return _read_schema_csv(path, CANDIDATES_COLUMNS)


def write_generations_csv(
    path: str | PathLike[str],
    rows: Sequence[Mapping[str, object]],
) -> Path:
    """Atomically write generation rows following the generations schema."""

    return _write_schema_csv(path, rows, GENERATIONS_COLUMNS)


def read_generations_csv(path: str | PathLike[str]) -> list[dict[str, str]]:
    """Read and schema-validate a generations CSV."""

    return _read_schema_csv(path, GENERATIONS_COLUMNS)


__all__ = [
    "ARCHIVE_SNAPSHOTS_DIRECTORY",
    "CANDIDATES_COLUMNS",
    "CANDIDATES_FILENAME",
    "DATASET_RECORD_FILENAME",
    "FINAL_ARCHIVE_FILENAME",
    "FINGERPRINT_PREFIX",
    "GENERATIONS_COLUMNS",
    "GENERATIONS_FILENAME",
    "INPUTS_DIRECTORY",
    "MAPPING_RECORD_FILENAME",
    "MMAP_MANIFEST_RECORD_FILENAME",
    "RUN_MANIFEST_FILENAME",
    "RUN_MANIFEST_FINGERPRINT_FILENAME",
    "RUN_MANIFEST_FORMAT",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RUN_STATES",
    "RUN_STATUS_FILENAME",
    "STATUS_FORMAT",
    "STATUS_SCHEMA_VERSION",
    "archive_snapshot_path",
    "build_dataset_record",
    "build_mapping_record",
    "build_mmap_manifest_record",
    "build_run_manifest",
    "build_status",
    "canonical_json_text",
    "feature_id",
    "fingerprint_bytes",
    "fingerprint_expression",
    "fingerprint_file",
    "fingerprint_mapping",
    "fingerprint_mmap_manifest",
    "fingerprint_text",
    "generation_from_snapshot_path",
    "load_run_manifest",
    "load_status",
    "read_candidates_csv",
    "read_generations_csv",
    "write_candidates_csv",
    "write_generations_csv",
    "write_run_manifest",
    "write_status",
]

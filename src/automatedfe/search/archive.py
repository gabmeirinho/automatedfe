"""The stateful archive step for multiobjective feature search."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.evaluation import Evaluator
from geneticengine.problems import Fitness, Problem
from geneticengine.problems.helpers import non_dominated
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation
from geneticengine.solutions.individual import PhenotypicIndividual

from ..features.feature_schema import code_lists_from_mapping
from ..features.grammar import NON_TERMINALS, TERMINALS, expr

logger = logging.getLogger(__name__)

FORMAT_IDENTIFIER = "automatedfe-archive"
FORMAT_VERSION = 1
ACTIVE_SET_FORMAT_IDENTIFIER = "automatedfe-active-set"
ACTIVE_SET_FORMAT_VERSION = 1
# Structured run snapshots never embed a label mapping: they reference the
# single run-level mapping owned by the run manifest instead. This format is
# distinct from the standalone archive format so loose outputs are never
# mistaken for structured run artifacts.
SNAPSHOT_FORMAT_IDENTIFIER = "automatedfe-archive-snapshot"
SNAPSHOT_FORMAT_VERSION = 1
SNAPSHOT_MAPPING_REFERENCE = {"file": "manifest.json", "source": "run_manifest"}
OBJECTIVES_PER_ARCHIVE = 4
ARCHIVE_PROXY_OBJECTIVES = 3
DEFAULT_ARCHIVE_QUALITY_THRESHOLD = 0.001
DEFAULT_ARCHIVE_CORRELATION_THRESHOLD = 0.85
DEFAULT_ACTIVE_CORRELATION_THRESHOLD = 0.90
DEFAULT_PROMOTION_INTERVAL = 5
DEFAULT_FIRST_PROMOTION_TOP_K = 2
DEFAULT_PROMOTION_ADD_K = 1
DEFAULT_PROMOTION_REFRESH_TOP_N = 50
DEFAULT_PROMOTION_MIN_GAIN = 0.0
DEFAULT_PROMOTION_MEAN_GAIN = 0.0005

_NODE_TYPES: dict[str, type] = {
    node_type.__name__: node_type
    for node_type in (*TERMINALS, *NON_TERMINALS)
}


def validate_correlation_threshold(threshold: float) -> float:
    """Validate and normalize an absolute-correlation threshold."""

    try:
        value = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("correlation threshold must be a finite number") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("correlation threshold must be between 0 and 1")
    return value


def validate_archive_quality_threshold(threshold: float) -> float:
    """Validate the minimum per-fold proxy improvement threshold."""

    try:
        value = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("archive quality threshold must be a finite number") from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("archive quality threshold must be finite and non-negative")
    return value


def _normalize_signal(signal: object) -> tuple[np.ndarray | None, str | None]:
    """Return a one-dimensional finite, non-constant signal or its rejection."""

    try:
        values = np.asarray(signal, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None, "signal_invalid"
    if values.size == 0:
        return None, "signal_empty"
    if not np.all(np.isfinite(values)):
        return None, "signal_nonfinite"
    if float(np.std(values)) <= 1e-12:
        return None, "signal_constant"
    return values, None


def absolute_pearson_correlation(signal_1: object, signal_2: object) -> float:
    """Return the absolute Pearson correlation of two valid signals.

    ``nan`` is returned when either signal is empty, non-finite, constant, or
    has a different length.  Returning a value instead of raising keeps
    candidate-local signal problems inside the archive filtering path.
    """

    first, first_reason = _normalize_signal(signal_1)
    second, second_reason = _normalize_signal(signal_2)
    if (
        first is None
        or second is None
        or first_reason is not None
        or second_reason is not None
        or first.size != second.size
    ):
        return float("nan")

    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator <= 1e-12:
        return float("nan")
    return min(
        1.0,
        max(0.0, abs(float(np.dot(first_centered, second_centered) / denominator))),
    )


def correlation_rejection(
    candidate: object,
    reference_signals: Sequence[object],
    threshold: float,
) -> dict[str, Any]:
    """Return deterministic diagnostics for candidate/reference correlation.

    The first reference reaching the inclusive threshold is reported.  When
    no reference reaches it, ``matched_index`` identifies the reference with
    the largest observed absolute correlation, which makes diagnostics stable
    without changing the admission decision.
    """

    threshold = validate_correlation_threshold(threshold)
    candidate_values, candidate_reason = _normalize_signal(candidate)
    if candidate_values is None:
        return {
            "rejected": True,
            "reason": candidate_reason or "signal_invalid",
            "abs_corr": float("nan"),
            "matched_index": None,
            "max_abs_corr": float("nan"),
        }

    maximum = float("nan")
    maximum_index: int | None = None
    for index, reference in enumerate(reference_signals):
        reference_values, reference_reason = _normalize_signal(reference)
        if reference_values is None:
            return {
                "rejected": True,
                "reason": reference_reason or "reference_invalid",
                "abs_corr": float("nan"),
                "matched_index": index,
                "max_abs_corr": maximum,
            }
        if reference_values.size != candidate_values.size:
            return {
                "rejected": True,
                "reason": "signal_shape_mismatch",
                "abs_corr": float("nan"),
                "matched_index": index,
                "max_abs_corr": maximum,
            }

        absolute_correlation = absolute_pearson_correlation(
            candidate_values,
            reference_values,
        )
        if not math.isfinite(absolute_correlation):
            return {
                "rejected": True,
                "reason": "correlation_invalid",
                "abs_corr": float("nan"),
                "matched_index": index,
                "max_abs_corr": maximum,
            }
        if maximum_index is None or absolute_correlation > maximum:
            maximum = absolute_correlation
            maximum_index = index
        # Centering and the norm-based implementation can be a few ulps below
        # one for a signal compared with itself.  Treat the public threshold
        # as inclusive while remaining deterministic.
        if absolute_correlation + 1e-12 >= threshold:
            return {
                "rejected": True,
                "reason": "pairwise_threshold",
                "abs_corr": absolute_correlation,
                "matched_index": index,
                "max_abs_corr": maximum,
            }

    return {
        "rejected": False,
        "reason": "",
        "abs_corr": maximum,
        "matched_index": maximum_index,
        "max_abs_corr": maximum,
    }


def is_correlated_pairwise(signal_1: object, signal_2: object, threshold: float) -> bool:
    """Return whether two signals have inclusive absolute correlation."""

    threshold = validate_correlation_threshold(threshold)
    correlation = absolute_pearson_correlation(signal_1, signal_2)
    return bool(
        math.isfinite(correlation) and correlation + 1e-12 >= threshold
    )


def encode_expression(node: expr) -> dict[str, object]:
    """Serialize a grammar expression into an allowlisted JSON structure."""

    node_type = type(node)
    if node_type not in _NODE_TYPES.values():
        raise TypeError(f"Unsupported expression node type: {node_type.__name__}")
    return {
        "type": node_type.__name__,
        "fields": {
            field.name: _encode_field(getattr(node, field.name))
            for field in fields(node)
        },
    }


def _encode_field(value: object) -> object:
    if isinstance(value, int):
        return value
    if isinstance(value, expr):
        return encode_expression(value)
    raise TypeError(f"Unsupported expression field of type {type(value).__name__}")


def decode_expression(data: object) -> expr:
    """Reconstruct a grammar expression from a serialized JSON structure."""

    if not isinstance(data, dict):
        raise TypeError(
            f"Serialized expression must be a JSON object, got {type(data).__name__}"
        )
    node_name = data.get("type")
    if not isinstance(node_name, str):
        raise TypeError("Serialized expression is missing its 'type' name")
    node_type = _NODE_TYPES.get(node_name)
    if node_type is None:
        raise ValueError(f"Unknown expression node type: {node_name!r}")
    fields_data = data.get("fields")
    if not isinstance(fields_data, dict):
        raise TypeError(
            f"Serialized {node_name} expression is missing its 'fields' object"
        )
    expected = {field.name for field in fields(node_type)}
    if set(fields_data) != expected:
        raise ValueError(
            f"{node_name} expects fields {sorted(expected)}, got {sorted(fields_data)}"
        )
    try:
        return node_type(
            **{name: _decode_field(value) for name, value in fields_data.items()}
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Cannot reconstruct {node_name}: {error}") from error


def _decode_field(value: object) -> object:
    if isinstance(value, bool):
        raise TypeError(f"Boolean is not a valid expression field: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return decode_expression(value)
    raise ValueError(f"Unsupported expression field value: {value!r}")


@dataclass(frozen=True)
class ArchiveSnapshot:
    """A loaded archive: reconstructed expressions and their objectives."""

    version: int
    minimize: tuple[bool, ...]
    mapping: Mapping[str, Mapping[str, int]]
    expressions: tuple[expr, ...]
    objectives: tuple[tuple[float, ...], ...]

    def __len__(self) -> int:
        return len(self.expressions)


@dataclass(frozen=True, slots=True)
class ActiveSetSnapshot:
    """A loaded active-set snapshot in promotion order.

    ``expressions`` and ``gains`` contain only the promoted candidates;
    ``promotion_events`` and ``promotion_checks`` retain the full auditable
    promotion history even when the final active set is empty.
    """

    version: int
    baseline_version: int
    minimize: tuple[bool, ...]
    mapping: Mapping[str, Mapping[str, int]]
    expressions: tuple[expr, ...]
    gains: tuple[tuple[float, ...], ...]
    promotion_generations: tuple[int | None, ...]
    promoted_baseline_versions: tuple[int | None, ...]
    promotion_events: tuple[dict[str, object], ...]
    promotion_checks: tuple[dict[str, object], ...]

    def __len__(self) -> int:
        return len(self.expressions)


def _resolve_mapping(
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None,
) -> Mapping[str, Mapping[str, int]]:
    """Resolve a mapping argument into a full label-mapping dict."""

    if mapping is None:
        from ..data.encoding import DEFAULT_MAPPING_OUTPUT, load_label_mapping

        return load_label_mapping(DEFAULT_MAPPING_OUTPUT)
    if isinstance(mapping, (str, PathLike)):
        from ..data.encoding import load_label_mapping

        return load_label_mapping(Path(mapping))
    return mapping


def _validate_mapping_matches(
    stored_mapping: Mapping[str, Mapping[str, int]],
    provided_mapping: Mapping[str, Mapping[str, int]],
) -> None:
    try:
        stored = code_lists_from_mapping(stored_mapping)
        provided = code_lists_from_mapping(provided_mapping)
    except ValueError as error:
        raise ValueError(f"Cannot compare label mappings: {error}") from error
    if stored != provided:
        raise ValueError(
            "Archive label mapping does not match the provided mapping"
        )


def _build_document(
    expressions: Sequence[expr],
    objectives: Sequence[Sequence[float]],
    *,
    minimize: Sequence[bool],
    mapping: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    return {
        "format": FORMAT_IDENTIFIER,
        "version": FORMAT_VERSION,
        "problem": {
            "number_of_objectives": len(minimize),
            "minimize": [bool(value) for value in minimize],
        },
        "mapping": {family: dict(values) for family, values in mapping.items()},
        "expressions": [
            {
                "expression": encode_expression(expression),
                "objectives": [float(value) for value in entry_objectives],
            }
            for expression, entry_objectives in zip(expressions, objectives)
        ],
    }


def _validate_entries(data: object, n_objectives: int) -> list[dict[str, object]]:
    if not isinstance(data, list):
        raise TypeError("Archive 'expressions' must be a JSON list")
    entries: list[dict[str, object]] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict) or set(entry) != {"expression", "objectives"}:
            raise ValueError(
                f"Archive entry {index} must contain exactly "
                "'expression' and 'objectives' keys"
            )
        objectives = entry["objectives"]
        if not isinstance(objectives, list) or len(objectives) != n_objectives:
            raise ValueError(
                f"Archive entry {index} must declare exactly "
                f"{n_objectives} objective values"
            )
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in objectives
        ):
            raise ValueError(
                f"Archive entry {index} must declare numeric objective values"
            )
        entries.append(entry)
    return entries


def _validate_document(
    data: object,
    path: Path,
) -> tuple[int, dict[str, object], dict[str, object], list[dict[str, object]]]:
    """Validate a loaded archive document; return (version, problem, mapping, entries)."""

    if not isinstance(data, dict):
        raise TypeError(f"Archive JSON must be an object: {path}")
    if data.get("format") != FORMAT_IDENTIFIER:
        raise ValueError(f"Unknown archive format: {data.get('format')!r}")
    version = data.get("version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported archive version {version!r} (expected {FORMAT_VERSION})"
        )
    problem = data.get("problem")
    if not isinstance(problem, dict):
        raise TypeError("Archive is missing its 'problem' metadata")
    n_objectives = problem.get("number_of_objectives")
    minimize = problem.get("minimize")
    if n_objectives != OBJECTIVES_PER_ARCHIVE:
        raise ValueError(
            f"Archive problem must declare {OBJECTIVES_PER_ARCHIVE} objectives"
        )
    if not isinstance(minimize, list) or len(minimize) != n_objectives:
        raise ValueError(
            f"Archive problem must declare exactly {n_objectives} "
            "minimization directions"
        )
    if not all(isinstance(value, bool) for value in minimize):
        raise ValueError("Archive minimization directions must be boolean values")
    mapping = data.get("mapping")
    if not isinstance(mapping, dict):
        raise TypeError("Archive is missing its 'mapping' metadata")
    try:
        code_lists_from_mapping(mapping)
    except ValueError as error:
        raise ValueError(f"Archive label mapping is invalid: {error}") from error
    entries = _validate_entries(data.get("expressions"), n_objectives)
    return version, problem, mapping, entries


def _atomic_write_json(path: Path, data: object) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
            json.dump(data, temp_file, indent=2, sort_keys=True)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise
    return path


def load_archive(
    path: str | PathLike[str],
    *,
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
) -> ArchiveSnapshot:
    """Load and validate an archive JSON snapshot.

    The label mapping embedded in the archive is validated against *mapping*
    when one is provided, so categorical expressions are only reconstructed
    when their code lists match. Loading never merges or resumes a
    search.
    """

    archive_path = Path(path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive JSON file does not exist: {archive_path}")
    try:
        with open(archive_path) as archive_file:
            data = json.load(archive_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Archive is not valid JSON: {archive_path}: {error}") from error

    version, problem, mapping_data, entries = _validate_document(data, archive_path)
    if mapping is not None:
        _validate_mapping_matches(mapping_data, _resolve_mapping(mapping))

    expressions = tuple(decode_expression(entry["expression"]) for entry in entries)
    objectives = tuple(tuple(entry["objectives"]) for entry in entries)
    return ArchiveSnapshot(
        version=version,
        minimize=tuple(problem["minimize"]),
        mapping=mapping_data,
        expressions=expressions,
        objectives=objectives,
    )


def _validate_active_set_entries(data: object) -> list[dict[str, object]]:
    if not isinstance(data, list):
        raise TypeError("Active-set 'expressions' must be a JSON list")
    allowed = {
        "expression",
        "gains",
        "promotion_generation",
        "promoted_baseline_version",
    }
    entries: list[dict[str, object]] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict) or "expression" not in entry:
            raise ValueError(
                f"Active-set entry {index} must contain an 'expression' key"
            )
        unknown = set(entry) - allowed
        if unknown:
            raise ValueError(
                f"Active-set entry {index} has unknown keys: {sorted(unknown)}"
            )
        for name in ("promotion_generation", "promoted_baseline_version"):
            value = entry.get(name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(
                    f"Active-set entry {index} must declare a non-negative "
                    f"integer {name}"
                )
        gains = entry.get("gains")
        if gains is not None and (
            not isinstance(gains, list)
            or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in gains
            )
        ):
            raise ValueError(
                f"Active-set entry {index} must declare numeric gain values"
            )
        entries.append(entry)
    return entries


def load_active_set_snapshot(
    path: str | PathLike[str],
    *,
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
) -> ActiveSetSnapshot:
    """Load and validate an active-set snapshot JSON file.

    The label mapping embedded in the snapshot is validated against *mapping*
    when one is provided. An empty active set loads normally: the returned
    snapshot has no expressions while still carrying the promotion history.
    """

    snapshot_path = Path(path).resolve()
    if not snapshot_path.is_file():
        raise FileNotFoundError(
            f"Active-set JSON file does not exist: {snapshot_path}"
        )
    try:
        with open(snapshot_path) as snapshot_file:
            data = json.load(snapshot_file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Active-set snapshot is not valid JSON: {snapshot_path}: {error}"
        ) from error

    if not isinstance(data, dict):
        raise TypeError(f"Active-set JSON must be an object: {snapshot_path}")
    if data.get("format") != ACTIVE_SET_FORMAT_IDENTIFIER:
        raise ValueError(
            f"Unknown active-set format: {data.get('format')!r}"
        )
    version = data.get("version")
    if version != ACTIVE_SET_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported active-set version {version!r} "
            f"(expected {ACTIVE_SET_FORMAT_VERSION})"
        )
    problem = data.get("problem")
    if not isinstance(problem, dict):
        raise TypeError("Active-set snapshot is missing its 'problem' metadata")
    n_objectives = problem.get("number_of_objectives")
    minimize = problem.get("minimize")
    if n_objectives != OBJECTIVES_PER_ARCHIVE:
        raise ValueError(
            f"Active-set problem must declare {OBJECTIVES_PER_ARCHIVE} objectives"
        )
    if not isinstance(minimize, list) or len(minimize) != n_objectives:
        raise ValueError(
            f"Active-set problem must declare exactly {n_objectives} "
            "minimization directions"
        )
    if not all(isinstance(value, bool) for value in minimize):
        raise ValueError("Active-set minimization directions must be boolean values")
    baseline_version = data.get("baseline_version")
    if isinstance(baseline_version, bool) or not isinstance(baseline_version, int):
        raise ValueError("Active-set baseline_version must be an integer")
    if baseline_version < 0:
        raise ValueError("Active-set baseline_version must be non-negative")

    mapping_data = data.get("mapping")
    if not isinstance(mapping_data, dict):
        raise TypeError("Active-set snapshot is missing its 'mapping' metadata")
    try:
        code_lists_from_mapping(mapping_data)
    except ValueError as error:
        raise ValueError(f"Active-set label mapping is invalid: {error}") from error
    if mapping is not None:
        _validate_mapping_matches(mapping_data, _resolve_mapping(mapping))

    entries = _validate_active_set_entries(data.get("expressions"))
    events = data.get("promotion_events")
    checks = data.get("promotion_checks")
    if not isinstance(events, list) or not isinstance(checks, list):
        raise TypeError(
            "Active-set snapshot must declare 'promotion_events' and "
            "'promotion_checks' lists"
        )

    expressions = tuple(decode_expression(entry["expression"]) for entry in entries)
    gains = tuple(
        tuple(float(value) for value in entry["gains"])
        if entry.get("gains") is not None
        else ()
        for entry in entries
    )
    promotion_generations = tuple(
        int(entry["promotion_generation"])
        if entry.get("promotion_generation") is not None
        else None
        for entry in entries
    )
    promoted_baseline_versions = tuple(
        int(entry["promoted_baseline_version"])
        if entry.get("promoted_baseline_version") is not None
        else None
        for entry in entries
    )
    return ActiveSetSnapshot(
        version=version,
        baseline_version=baseline_version,
        minimize=tuple(minimize),
        mapping=mapping_data,
        expressions=expressions,
        gains=gains,
        promotion_generations=promotion_generations,
        promoted_baseline_versions=promoted_baseline_versions,
        promotion_events=tuple(dict(event) for event in events),
        promotion_checks=tuple(dict(check) for check in checks),
    )


def _validate_snapshot_mapping_ref(mapping_ref: object) -> dict[str, str]:
    if not isinstance(mapping_ref, Mapping):
        raise TypeError(
            "Snapshot mapping_ref must be a JSON object identifying the run manifest"
        )
    if set(mapping_ref) != {"file", "source"}:
        raise ValueError(
            "Snapshot mapping_ref must declare exactly 'file' and 'source' keys"
        )
    resolved: dict[str, str] = {}
    for name in ("file", "source"):
        value = mapping_ref[name]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Snapshot mapping_ref {name!r} must be a non-empty string"
            )
        resolved[name] = value
    return resolved


def build_snapshot_document(
    expressions: Sequence[expr],
    objectives: Sequence[Sequence[float]],
    *,
    minimize: Sequence[bool],
    mapping_ref: Mapping[str, str],
) -> dict[str, object]:
    """Build a structured run snapshot without embedding a mapping copy.

    Structured snapshots reference the run manifest owning the single run-level
    label mapping; *mapping_ref* names that manifest. This keeps the standalone
    archive serialization (which embeds its mapping) untouched.
    """

    reference = _validate_snapshot_mapping_ref(mapping_ref)
    return {
        "format": SNAPSHOT_FORMAT_IDENTIFIER,
        "version": SNAPSHOT_FORMAT_VERSION,
        "problem": {
            "number_of_objectives": len(minimize),
            "minimize": [bool(value) for value in minimize],
        },
        "mapping_ref": reference,
        "expressions": [
            {
                "expression": encode_expression(expression),
                "objectives": [float(value) for value in entry_objectives],
            }
            for expression, entry_objectives in zip(expressions, objectives)
        ],
    }


def write_snapshot(
    path: str | PathLike[str],
    expressions: Sequence[expr],
    objectives: Sequence[Sequence[float]],
    *,
    minimize: Sequence[bool],
    mapping_ref: Mapping[str, str],
) -> Path:
    """Atomically write a structured run snapshot JSON file."""

    document = build_snapshot_document(
        expressions,
        objectives,
        minimize=minimize,
        mapping_ref=mapping_ref,
    )
    return _atomic_write_json(Path(path).resolve(), document)


def load_snapshot(
    path: str | PathLike[str],
    mapping: Mapping[str, Mapping[str, int]] | None,
) -> ArchiveSnapshot:
    """Load and validate a structured run snapshot.

    A structured snapshot never embeds a label mapping: *mapping* must be the
    single run-level mapping recorded in the run manifest, which is validated
    against the snapshot's ``mapping_ref``. Loading never merges or resumes a
    search.
    """

    snapshot_path = Path(path).resolve()
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Snapshot JSON file does not exist: {snapshot_path}")
    try:
        with open(snapshot_path) as snapshot_file:
            data = json.load(snapshot_file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Snapshot is not valid JSON: {snapshot_path}: {error}"
        ) from error

    if not isinstance(data, dict):
        raise TypeError(f"Snapshot JSON must be an object: {snapshot_path}")
    if data.get("format") != SNAPSHOT_FORMAT_IDENTIFIER:
        raise ValueError(
            f"Unknown snapshot format: {data.get('format')!r}"
        )
    version = data.get("version")
    if version != SNAPSHOT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported snapshot version {version!r} "
            f"(expected {SNAPSHOT_FORMAT_VERSION})"
        )
    if "mapping" in data:
        raise ValueError(
            "Structured snapshots must not embed a mapping copy; resolve the "
            "run-level mapping from the run manifest"
        )
    if data.get("mapping_ref") is None:
        raise ValueError(
            "Snapshot is missing its 'mapping_ref' reference to the run manifest"
        )
    _validate_snapshot_mapping_ref(data.get("mapping_ref"))

    problem = data.get("problem")
    if not isinstance(problem, dict):
        raise TypeError("Snapshot is missing its 'problem' metadata")
    n_objectives = problem.get("number_of_objectives")
    minimize = problem.get("minimize")
    if n_objectives != OBJECTIVES_PER_ARCHIVE:
        raise ValueError(
            f"Snapshot problem must declare {OBJECTIVES_PER_ARCHIVE} objectives"
        )
    if not isinstance(minimize, list) or len(minimize) != n_objectives:
        raise ValueError(
            f"Snapshot problem must declare exactly {n_objectives} "
            "minimization directions"
        )
    if not all(isinstance(value, bool) for value in minimize):
        raise ValueError("Snapshot minimization directions must be boolean values")

    entries = _validate_entries(data.get("expressions"), n_objectives)
    if mapping is None:
        raise ValueError(
            "Structured snapshots resolve the run-level mapping; a mapping is "
            "required to load this snapshot"
        )
    try:
        code_lists_from_mapping(mapping)
    except ValueError as error:
        raise ValueError(
            f"Run-level mapping cannot resolve snapshot expressions: {error}"
        ) from error

    expressions = tuple(decode_expression(entry["expression"]) for entry in entries)
    objectives = tuple(tuple(entry["objectives"]) for entry in entries)
    return ArchiveSnapshot(
        version=version,
        minimize=tuple(minimize),
        mapping=dict(mapping),
        expressions=expressions,
        objectives=objectives,
    )


class ArchiveStep(GeneticStep):
    """Maintain one global archive while transparently passing populations on.

    The step owns the archive state. For each complete input population it:

    1. evaluates the complete population;
    2. removes invalid and duplicate expressions;
    3. combines the candidates with the previous archive;
    4. delegates front calculation to Genetic Engine's ``non_dominated``; and
    5. appends current candidates that are present on the resulting global
       front, retaining every previously admitted archive member.

    Every evaluated population member is yielded unchanged so this step can be
    appended to a Genetic Engine ``SequenceStep`` without changing evolution.

    When *archive_path* is provided, an atomic JSON snapshot of the permanent
    archive is written after every generation. *mapping* supplies the label
    mapping embedded in those snapshots; it defaults to the persisted
    preprocessing mapping.
    """

    def __init__(
        self,
        *,
        archive_path: str | PathLike[str] | None = None,
        mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    ) -> None:
        self.archive: list[PhenotypicIndividual] = []
        self._problem: Problem | None = None
        self._mapping = mapping
        self.archive_path: Path | None = (
            Path(archive_path).resolve() if archive_path is not None else None
        )

    @property
    def individuals(self) -> tuple[PhenotypicIndividual, ...]:
        """Return the live archived individuals in stable order."""

        return tuple(self.archive)

    def __len__(self) -> int:
        return len(self.archive)

    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random: RandomSource,
        population: Iterator[PhenotypicIndividual],
        target_size: int,
        generation: int,
    ) -> Iterator[PhenotypicIndividual]:
        self._validate_problem(problem)
        self._problem = problem
        evaluated = list(evaluator.evaluate(problem, population))
        current = self._valid_unique(evaluated, problem)
        self._append_archive_admissions(current, problem)
        if self.archive_path is not None:
            self.save(self.archive_path)
        yield from evaluated

    def _append_archive_admissions(
        self,
        current: Sequence[PhenotypicIndividual],
        problem: Problem,
    ) -> list[PhenotypicIndividual]:
        """Append current candidates that survive the combined Pareto gate."""

        candidates = self._deduplicate([*self.archive, *current])
        front = list(non_dominated(iter(candidates), problem))
        front_keys = {self._expression_key(individual) for individual in front}
        archive_keys = {
            self._expression_key(individual) for individual in self.archive
        }
        for individual in current:
            key = self._expression_key(individual)
            if key in front_keys and key not in archive_keys:
                self.archive.append(individual)
                archive_keys.add(key)
        return front

    def reevaluate_archive(
        self,
        problem: Problem,
        evaluator: Evaluator,
    ) -> None:
        """Re-score the complete archive without changing membership or order.

        This is required when the fitness baseline changes: objective values
        calculated against different baselines must never share a Pareto
        comparison. Fitness is explicitly removed so the refresh does not
        depend on evaluator-specific cache invalidation. If a refresh produces
        invalid fitness, the member keeps its last finite objective vector so
        the permanent archive remains serializable.
        """

        self._validate_problem(problem)
        self._problem = problem
        archived = list(self.archive)
        previous_objectives = {
            self._expression_key(individual): objectives
            for individual in archived
            if (objectives := self._finite_objectives(individual, problem)) is not None
        }
        for individual in archived:
            individual.fitness_store.pop(problem, None)
        evaluated = list(evaluator.evaluate(problem, iter(archived)))
        valid = self._valid_unique(evaluated, problem)
        refreshed = {
            self._expression_key(individual): individual for individual in valid
        }

        for individual in archived:
            key = self._expression_key(individual)
            refreshed_individual = refreshed.get(key)
            if refreshed_individual is not None:
                if refreshed_individual is not individual:
                    objectives = self._finite_objectives(refreshed_individual, problem)
                    assert objectives is not None
                    individual.set_fitness(
                        problem,
                        Fitness(list(objectives), valid=True),
                    )
                continue

            objectives = previous_objectives.get(key)
            if objectives is None:
                raise ValueError(
                    "Cannot refresh archive member without finite objective evidence: "
                    f"{key}"
                )
            individual.set_fitness(problem, Fitness(list(objectives), valid=True))

    def save(
        self,
        path: str | PathLike[str] | None = None,
        *,
        mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    ) -> Path:
        """Write an atomic JSON snapshot of the permanent archive.

        *mapping* defaults to the mapping configured for the step, then to the
        persisted default.
        """

        save_path = self._resolve_save_path(path)
        if self._problem is None:
            raise ValueError("Cannot save an archive that has not evaluated a population")
        resolved_mapping = mapping if mapping is not None else self._mapping
        document = _build_document(
            expressions=[individual.get_phenotype() for individual in self.archive],
            objectives=[
                tuple(
                    float(value)
                    for value in individual.get_fitness(self._problem).fitness_components
                )
                for individual in self.archive
            ],
            minimize=self._problem.minimize,
            mapping=_resolve_mapping(resolved_mapping),
        )
        return _atomic_write_json(save_path, document)

    def archive_snapshot(self) -> dict[str, object]:
        """Return the current front as a mapping-free structured snapshot.

        Structured run snapshots never embed the label mapping: they reference
        the single run-level mapping owned by the run manifest via
        ``mapping_ref``. This document can be reloaded with
        :func:`load_snapshot` when the run-level mapping is provided.
        """

        if self._problem is None:
            raise ValueError(
                "Cannot snapshot an archive that has not evaluated a population"
            )
        return build_snapshot_document(
            expressions=[
                individual.get_phenotype() for individual in self.archive
            ],
            objectives=[
                tuple(
                    float(value)
                    for value in individual.get_fitness(self._problem).fitness_components
                )
                for individual in self.archive
            ],
            minimize=self._problem.minimize,
            mapping_ref=SNAPSHOT_MAPPING_REFERENCE,
        )

    def _resolve_save_path(self, path: str | PathLike[str] | None) -> Path:
        if path is None:
            if self.archive_path is None:
                raise ValueError(
                    "No archive path configured; pass an explicit path to save()"
                )
            return self.archive_path
        return Path(path).resolve()

    @staticmethod
    def _validate_problem(problem: Problem) -> None:
        if problem.number_of_objectives() != 4:
            raise ValueError("ArchiveStep requires four objectives")

    @classmethod
    def _valid_unique(
        cls,
        individuals: Sequence[PhenotypicIndividual],
        problem: Problem,
    ) -> list[PhenotypicIndividual]:
        valid: list[PhenotypicIndividual] = []
        seen: set[str] = set()
        for individual in individuals:
            fitness = individual.get_fitness(problem)
            key = cls._expression_key(individual)
            if not fitness.valid or len(fitness.fitness_components) != problem.number_of_objectives():
                cls._log_invalid(key, fitness.fitness_components)
                continue
            try:
                finite = all(
                    math.isfinite(float(value))
                    for value in fitness.fitness_components
                )
            except (TypeError, ValueError):
                finite = False
            if not finite:
                cls._log_invalid(key, fitness.fitness_components)
                continue
            if key in seen:
                continue
            seen.add(key)
            valid.append(individual)
        return valid

    @staticmethod
    def _finite_objectives(
        individual: PhenotypicIndividual,
        problem: Problem,
    ) -> tuple[float, ...] | None:
        """Return finite valid objectives for an individual, if available."""

        try:
            fitness = individual.get_fitness(problem)
            if (
                not fitness.valid
                or len(fitness.fitness_components) != problem.number_of_objectives()
            ):
                return None
            objectives = tuple(float(value) for value in fitness.fitness_components)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(value) for value in objectives):
            return None
        return objectives

    @classmethod
    def _deduplicate(
        cls,
        individuals: Sequence[PhenotypicIndividual],
    ) -> list[PhenotypicIndividual]:
        unique: list[PhenotypicIndividual] = []
        seen: set[str] = set()
        for individual in individuals:
            key = cls._expression_key(individual)
            if key in seen:
                continue
            seen.add(key)
            unique.append(individual)
        return unique

    @staticmethod
    def _expression_key(individual: PhenotypicIndividual) -> str:
        from .search import canonical_expression_key

        return canonical_expression_key(individual.get_phenotype())

    @staticmethod
    def _log_invalid(key: str, objectives: object) -> None:
        logger.warning(
            "Skipping invalid archive candidate %s with objectives=%r",
            key,
            objectives,
        )


class _ActiveSetManagerBase(ArchiveStep):
    """Shared implementation for active-set history and promotion state.

    For every generation, the complete evaluated population is first reduced
    to its four-objective Pareto front.  That generation front is then
    filtered in this order:

    1. minimum improvement on all three proxy folds;
    2. absolute Pearson correlation against every previously admitted history
       signal; and
    3. greedy same-generation peer clustering, retaining the best candidate in
       each correlated cluster.

    The final ``archive`` retains every candidate admitted by the global
    Pareto gate, while ``history`` is unbounded, structurally unique, and
    ordered by admission. Admission objectives are copied into
    ``admission_objectives`` so a later evaluator may refresh an individual's
    live fitness without changing the values that justified history admission.

    ``signal_provider`` receives a candidate phenotype and must return its
    full training-row signal.
    """

    def __init__(
        self,
        *,
        archive_path: str | PathLike[str] | None = None,
        mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
        signal_provider: Callable[[object], object] | None = None,
        archive_quality_threshold: float = DEFAULT_ARCHIVE_QUALITY_THRESHOLD,
        archive_correlation_threshold: float = DEFAULT_ARCHIVE_CORRELATION_THRESHOLD,
        use_active_set: bool = False,
        promotion_interval: int = DEFAULT_PROMOTION_INTERVAL,
        first_promotion_top_k: int = DEFAULT_FIRST_PROMOTION_TOP_K,
        promotion_add_k: int = DEFAULT_PROMOTION_ADD_K,
        promotion_refresh_top_n: int = DEFAULT_PROMOTION_REFRESH_TOP_N,
        active_correlation_threshold: float = DEFAULT_ACTIVE_CORRELATION_THRESHOLD,
        promotion_min_gain: float = DEFAULT_PROMOTION_MIN_GAIN,
        promotion_mean_gain: float = DEFAULT_PROMOTION_MEAN_GAIN,
    ) -> None:
        super().__init__(archive_path=archive_path, mapping=mapping)
        self.archive_quality_threshold = validate_archive_quality_threshold(
            archive_quality_threshold
        )
        self.archive_correlation_threshold = validate_correlation_threshold(
            archive_correlation_threshold
        )
        if not isinstance(use_active_set, bool):
            raise ValueError("use_active_set must be a boolean")
        for name, value in (
            ("promotion_interval", promotion_interval),
            ("first_promotion_top_k", first_promotion_top_k),
            ("promotion_add_k", promotion_add_k),
            ("promotion_refresh_top_n", promotion_refresh_top_n),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if promotion_interval == 0:
            raise ValueError("promotion_interval must be positive")
        try:
            promotion_min_gain = float(promotion_min_gain)
            promotion_mean_gain = float(promotion_mean_gain)
        except (TypeError, ValueError) as error:
            raise ValueError("promotion gain thresholds must be finite numbers") from error
        if not math.isfinite(promotion_min_gain) or not math.isfinite(promotion_mean_gain):
            raise ValueError("promotion gain thresholds must be finite numbers")
        self.signal_provider = signal_provider
        self.use_active_set = use_active_set
        self.promotion_interval = promotion_interval
        self.first_promotion_top_k = first_promotion_top_k
        self.promotion_add_k = promotion_add_k
        self.promotion_refresh_top_n = promotion_refresh_top_n
        self.active_correlation_threshold = validate_correlation_threshold(
            active_correlation_threshold
        )
        self.promotion_min_gain = promotion_min_gain
        self.promotion_mean_gain = promotion_mean_gain
        self.history: list[PhenotypicIndividual] = []
        self.admission_objectives: dict[str, tuple[float, ...]] = {}
        self._history_signals: list[np.ndarray] = []
        self._history_keys: set[str] = set()
        self.admission_diagnostics: list[dict[str, object]] = []
        self.generation_diagnostics: list[dict[str, object]] = []
        self.active_individuals: list[PhenotypicIndividual] = []
        self.baseline_version = 0
        self.promotion_events: list[dict[str, object]] = []
        self.promotion_checks: list[dict[str, object]] = []

    @property
    def history_individuals(self) -> tuple[PhenotypicIndividual, ...]:
        """Return history candidates in immutable admission order."""

        return tuple(self.history)

    @property
    def history_objectives(self) -> tuple[tuple[float, ...], ...]:
        """Return immutable admission objectives aligned with ``history``."""

        return tuple(
            self.admission_objectives[self._expression_key(individual)]
            for individual in self.history
            if self._expression_key(individual) in self.admission_objectives
        )

    @property
    def history_signals(self) -> tuple[np.ndarray, ...]:
        """Return copies of the signals used for history correlation checks."""

        return tuple(signal.copy() for signal in self._history_signals)

    @property
    def history_keys(self) -> frozenset[str]:
        """Return structural identities admitted to history."""

        return frozenset(self._history_keys)

    @property
    def active_keys(self) -> frozenset[str]:
        """Return structural identities in promotion order."""

        return frozenset(
            self._expression_key(individual) for individual in self.active_individuals
        )

    def maybe_promote(
        self,
        problem: Problem,
        generation: int,
        evaluator: Evaluator | None = None,
    ) -> bool:
        """Promote complementary history candidates at an interval boundary.

        Winners are selected one at a time.  Every winner advances
        ``baseline_version`` before the next candidate is scored, so the
        evaluator cannot accidentally compare a later winner with a stale
        active baseline.
        """

        if (
            not self.use_active_set
            or generation <= 0
            or generation % self.promotion_interval != 0
            or not self.history
        ):
            return False

        if evaluator is not None and self.promotion_refresh_top_n:
            self._refresh_history(evaluator, problem)

        requested = (
            self.first_promotion_top_k
            if not self.active_individuals
            else self.promotion_add_k
        )
        if requested <= 0:
            return False

        promoted = 0
        phase = "first_promotion" if not self.active_individuals else "incremental_promotion"
        for _ in range(requested):
            winner = self._select_promotion_winner(
                problem,
                generation,
                evaluator,
                phase=phase,
            )
            if winner is None:
                break
            candidate, gains, check_index = winner
            self.promotion_checks[check_index]["outcome"] = "promoted"
            self.promotion_checks[check_index]["reason"] = (
                "first_promotion_winner"
                if phase == "first_promotion"
                else "incremental_promotion_winner"
            )
            self.active_individuals.append(candidate)
            self.baseline_version += 1
            candidate.metadata["promoted_baseline_version"] = self.baseline_version
            candidate.metadata["promotion_generation"] = generation
            candidate.metadata["promotion_gains"] = [float(value) for value in gains]
            promoted += 1
            logger.info(
                "Promoted active candidate %s at generation %d: min_gain=%.6f mean_gain=%.6f",
                candidate.get_phenotype(),
                generation,
                min(gains),
                float(np.mean(gains)),
            )

        if promoted:
            self.promotion_events.append(
                {
                    "generation": generation,
                    "phase": phase,
                    "baseline_version": self.baseline_version,
                    "active_size": len(self.active_individuals),
                    "promoted_count": promoted,
                    "promotion_interval": self.promotion_interval,
                }
            )
        return promoted > 0

    def save_history(
        self,
        path: str | PathLike[str],
        *,
        mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    ) -> Path:
        """Write the complete filtered history to an atomic JSON snapshot.

        The document reuses the archive representation so the snapshot can be
        reloaded with :func:`load_archive`. Objectives are the immutable
        admission values, never later baseline refreshes.
        """

        save_path = Path(path).resolve()
        if self._problem is None:
            raise ValueError(
                "Cannot save a history that has not processed a population"
            )
        resolved_mapping = mapping if mapping is not None else self._mapping
        document = _build_document(
            expressions=[
                individual.get_phenotype() for individual in self.history
            ],
            objectives=[
                self._admission_objectives_for(index)
                for index in range(len(self.history))
            ],
            minimize=self._problem.minimize,
            mapping=_resolve_mapping(resolved_mapping),
        )
        return _atomic_write_json(save_path, document)

    def save_active_snapshot(
        self,
        path: str | PathLike[str],
        *,
        mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
    ) -> Path:
        """Write the promoted active set with its promotion history.

        Each entry stores the promotion-time split gains, the generation it
        was promoted at, and the baseline version it was scored against.
        ``promotion_events`` and ``promotion_checks`` retain the complete
        auditable history, so an empty requested snapshot is written without
        errors.
        """

        save_path = Path(path).resolve()
        if self._problem is None:
            raise ValueError(
                "Cannot save an active set that has not processed a population"
            )
        resolved_mapping = mapping if mapping is not None else self._mapping
        entries: list[dict[str, object]] = []
        for individual in self.active_individuals:
            key = self._expression_key(individual)
            metadata = individual.metadata
            gains = metadata.get("promotion_gains")
            if gains is None:
                gains = self.admission_objectives.get(key, ())
            entries.append(
                {
                    "expression": encode_expression(individual.get_phenotype()),
                    "gains": [float(value) for value in gains],
                    "promotion_generation": metadata.get("promotion_generation"),
                    "promoted_baseline_version": metadata.get(
                        "promoted_baseline_version"
                    ),
                }
            )
        document: dict[str, object] = {
            "format": ACTIVE_SET_FORMAT_IDENTIFIER,
            "version": ACTIVE_SET_FORMAT_VERSION,
            "problem": {
                "number_of_objectives": OBJECTIVES_PER_ARCHIVE,
                "minimize": [bool(value) for value in self._problem.minimize],
            },
            "mapping": {
                family: dict(values)
                for family, values in _resolve_mapping(resolved_mapping).items()
            },
            "baseline_version": self.baseline_version,
            "expressions": entries,
            "promotion_events": [dict(event) for event in self.promotion_events],
            "promotion_checks": [dict(check) for check in self.promotion_checks],
            "active_size": len(self.active_individuals),
        }
        return _atomic_write_json(save_path, document)

    def _refresh_history(self, evaluator: Evaluator, problem: Problem) -> None:
        """Refresh a bounded proxy shortlist before the exact promotion scan."""

        ranked = self._history_in_proxy_order()
        ranked = ranked[: min(self.promotion_refresh_top_n, len(ranked))]
        if ranked:
            list(evaluator.evaluate(problem, [self.history[index] for index in ranked]))

    def _history_in_proxy_order(self) -> list[int]:
        return sorted(
            range(len(self.history)),
            key=lambda index: min(
                self._admission_objectives_for(index)[:ARCHIVE_PROXY_OBJECTIVES]
            ),
            reverse=True,
        )

    def _admission_objectives_for(self, index: int) -> tuple[float, ...]:
        individual = self.history[index]
        key = self._expression_key(individual)
        if key in self.admission_objectives:
            return self.admission_objectives[key]
        if self._problem is None:
            return (0.0, 0.0, 0.0, 0.0)
        return tuple(
            float(value) for value in individual.get_fitness(self._problem).fitness_components
        )

    def _history_signal_for(self, index: int) -> np.ndarray | None:
        if index < len(self._history_signals):
            return self._history_signals[index]
        signal, _reason = self._signal_for(self.history[index])
        return signal

    def _active_signals(self) -> list[np.ndarray]:
        signals: list[np.ndarray] = []
        for individual in self.active_individuals:
            key = self._expression_key(individual)
            signal = None
            for index, history_individual in enumerate(self.history):
                if self._expression_key(history_individual) == key:
                    signal = self._history_signal_for(index)
                    break
            if signal is None:
                signal, _reason = self._signal_for(individual)
            if signal is not None:
                signals.append(signal)
        return signals

    def _select_promotion_winner(
        self,
        problem: Problem,
        generation: int,
        evaluator: Evaluator | None,
        *,
        phase: str,
    ) -> tuple[PhenotypicIndividual, tuple[float, ...], int] | None:
        active_keys = self.active_keys
        active_signals = self._active_signals()
        best: tuple[tuple[float, float, float], int, PhenotypicIndividual, tuple[float, ...]] | None = None

        for proxy_rank, index in enumerate(self._history_in_proxy_order(), start=1):
            candidate = self.history[index]
            key = self._expression_key(candidate)
            if key in active_keys:
                continue
            signal = self._history_signal_for(index)
            if signal is None:
                self._record_promotion_check(
                    candidate, generation, phase, proxy_rank, (), "rejected", "signal_invalid"
                )
                continue
            correlation = correlation_rejection(
                signal,
                active_signals,
                self.active_correlation_threshold,
            )
            if correlation["rejected"]:
                self._record_promotion_check(
                    candidate,
                    generation,
                    phase,
                    proxy_rank,
                    (),
                    "rejected",
                    "active_correlation",
                    correlation=correlation.get("abs_corr"),
                )
                continue

            if evaluator is not None:
                evaluated = list(evaluator.evaluate(problem, [candidate]))
                if not evaluated:
                    continue
                candidate = evaluated[0]
            try:
                fitness = candidate.get_fitness(problem)
                values = tuple(float(value) for value in fitness.fitness_components)
                gains = values[:ARCHIVE_PROXY_OBJECTIVES]
                cost = values[ARCHIVE_PROXY_OBJECTIVES]
                valid = (
                    fitness.valid
                    and len(values) == OBJECTIVES_PER_ARCHIVE
                    and all(math.isfinite(value) for value in values)
                )
            except (AttributeError, TypeError, ValueError, IndexError):
                gains = ()
                cost = float("inf")
                valid = False

            check_index = self._record_promotion_check(
                candidate,
                generation,
                phase,
                proxy_rank,
                gains,
                "checked" if valid else "rejected",
                "" if valid else "invalid_fitness",
            )
            if not valid:
                continue
            minimum = min(gains)
            mean = float(np.mean(gains))
            if minimum < self.promotion_min_gain or mean < self.promotion_mean_gain:
                self.promotion_checks[check_index]["outcome"] = "rejected"
                self.promotion_checks[check_index]["reason"] = "promotion_threshold"
                continue
            rank = (minimum, mean, -cost)
            if best is None or rank > best[0]:
                if best is not None:
                    self.promotion_checks[best[1]]["outcome"] = "not_selected"
                    self.promotion_checks[best[1]]["reason"] = "not_best_exact"
                best = (rank, check_index, candidate, gains)
            else:
                self.promotion_checks[check_index]["outcome"] = "not_selected"
                self.promotion_checks[check_index]["reason"] = "not_best_exact"
        if best is None:
            return None
        return best[2], best[3], best[1]

    def _record_promotion_check(
        self,
        individual: PhenotypicIndividual,
        generation: int,
        phase: str,
        proxy_rank: int,
        gains: Sequence[float],
        outcome: str,
        reason: str,
        *,
        correlation: object = None,
    ) -> int:
        admission = self.admission_objectives.get(self._expression_key(individual), ())
        record: dict[str, object] = {
            "generation": generation,
            "phase": phase,
            "baseline_version": self.baseline_version,
            "expression": str(individual.get_phenotype()),
            "proxy_rank": proxy_rank,
            "proxy_gains": [float(value) for value in admission[:ARCHIVE_PROXY_OBJECTIVES]],
            "current_gains": [float(value) for value in gains],
            "minimum_gain_threshold": self.promotion_min_gain,
            "mean_gain_threshold": self.promotion_mean_gain,
            "active_correlation_threshold": self.active_correlation_threshold,
            "outcome": outcome,
            "reason": reason,
        }
        if correlation is not None:
            value = float(correlation)
            record["abs_corr"] = value if math.isfinite(value) else None
        self.promotion_checks.append(record)
        return len(self.promotion_checks) - 1

    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random: RandomSource,
        population: Iterator[PhenotypicIndividual],
        target_size: int,
        generation: int,
    ) -> Iterator[PhenotypicIndividual]:
        self._validate_problem(problem)
        self._problem = problem
        evaluated = list(evaluator.evaluate(problem, population))
        self.process_evaluated_population(problem, evaluated, generation)
        if self.archive_path is not None:
            self.save(self.archive_path)
        yield from evaluated

    def process_evaluated_population(
        self,
        problem: Problem,
        population_list: Sequence[PhenotypicIndividual],
        generation: int,
    ) -> list[PhenotypicIndividual]:
        """Process an already evaluated generation and return its front.

        The explicit method is useful for deterministic active-set tests. The
        population is already evaluated by definition.
        """

        self._validate_problem(problem)
        self._problem = problem
        evaluated = list(population_list)
        valid = self._valid_unique_with_diagnostics(evaluated, problem, generation)

        # The archive admission gate combines historical and current
        # candidates, while the history admission front is deliberately
        # calculated from this generation alone. The latter admits useful
        # candidates that may later leave the global Pareto front.
        self._append_archive_admissions(valid, problem)
        generation_front = list(non_dominated(iter(valid), problem))
        generation_front_keys = {
            self._expression_key(individual) for individual in generation_front
        }
        for individual in valid:
            key = self._expression_key(individual)
            if key not in generation_front_keys:
                self._record_diagnostic(
                    generation=generation,
                    individual=individual,
                    outcome="not_pareto",
                    reason="dominated_in_generation",
                )

        self._admit_generation(generation_front, problem, generation)
        self.generation_diagnostics.append(
            {
                "generation": generation,
                "population_size": len(evaluated),
                "valid_size": len(valid),
                "front_size": len(generation_front),
                "history_size": len(self.history),
                "archive_size": len(self.archive),
            }
        )
        return generation_front

    def _valid_unique_with_diagnostics(
        self,
        individuals: Sequence[PhenotypicIndividual],
        problem: Problem,
        generation: int,
    ) -> list[PhenotypicIndividual]:
        valid: list[PhenotypicIndividual] = []
        seen: set[str] = set()
        for individual in individuals:
            key = self._expression_key(individual)
            try:
                fitness = individual.get_fitness(problem)
                components = tuple(fitness.fitness_components)
                if not fitness.valid:
                    raise ValueError("fitness_invalid")
                if len(components) != OBJECTIVES_PER_ARCHIVE:
                    raise ValueError("wrong_objective_count")
                objective_values = tuple(float(value) for value in components)
                if not all(math.isfinite(value) for value in objective_values):
                    raise ValueError("objectives_nonfinite")
            except (ArithmeticError, TypeError, ValueError, AttributeError) as error:
                reason = str(error) if str(error) in {
                    "fitness_invalid",
                    "wrong_objective_count",
                    "objectives_nonfinite",
                } else "invalid_fitness"
                self._record_diagnostic(
                    generation=generation,
                    individual=individual,
                    outcome="invalid",
                    reason=reason,
                )
                self._log_invalid(key, getattr(locals().get("fitness", None), "fitness_components", ()))
                continue

            if key in seen:
                self._record_diagnostic(
                    generation=generation,
                    individual=individual,
                    outcome="duplicate",
                    reason="duplicate_generation_expression",
                    objectives=objective_values,
                )
                continue
            seen.add(key)
            individual.metadata.setdefault(
                "evaluated_baseline_version", self.baseline_version
            )
            valid.append(individual)
        return valid

    def _admit_generation(
        self,
        generation_front: Sequence[PhenotypicIndividual],
        problem: Problem,
        generation: int,
    ) -> None:
        candidates: list[tuple[PhenotypicIndividual, np.ndarray, str, tuple[float, ...]]] = []
        for individual in generation_front:
            key = self._expression_key(individual)
            objectives = tuple(
                float(value)
                for value in individual.get_fitness(problem).fitness_components
            )
            if key in self._history_keys:
                self._record_diagnostic(
                    generation=generation,
                    individual=individual,
                    outcome="duplicate",
                    reason="duplicate_history_expression",
                    objectives=objectives,
                )
                continue

            proxy_minimum = min(objectives[:ARCHIVE_PROXY_OBJECTIVES])
            if proxy_minimum < self.archive_quality_threshold:
                self._record_diagnostic(
                    generation=generation,
                    individual=individual,
                    outcome="rejected",
                    reason="quality_threshold",
                    objectives=objectives,
                    proxy_minimum=proxy_minimum,
                )
                continue

            signal, signal_reason = self._signal_for(individual)
            if signal is None:
                self._record_diagnostic(
                    generation=generation,
                    individual=individual,
                    outcome="invalid",
                    reason=signal_reason or "signal_invalid",
                    objectives=objectives,
                    proxy_minimum=proxy_minimum,
                )
                continue

            historical = correlation_rejection(
                signal,
                self._history_signals,
                self.archive_correlation_threshold,
            )
            if historical["rejected"]:
                self._record_diagnostic(
                    generation=generation,
                    individual=individual,
                    outcome="rejected",
                    reason=f"historical_{historical['reason']}",
                    objectives=objectives,
                    proxy_minimum=proxy_minimum,
                    correlation=historical.get("abs_corr"),
                    matched_index=historical.get("matched_index"),
                )
                continue
            candidates.append((individual, signal, key, objectives))

        clusters: list[dict[str, object]] = []
        for individual, signal, key, objectives in candidates:
            placed = False
            for cluster_index, cluster in enumerate(clusters):
                peer_check = correlation_rejection(
                    signal,
                    cluster["signals"],  # type: ignore[arg-type]
                    self.archive_correlation_threshold,
                )
                if not peer_check["rejected"]:
                    continue

                placed = True
                representative = cluster["representative"]
                representative_objectives = cluster["objectives"]
                candidate_rank = self._rank_key(objectives)
                representative_rank = self._rank_key(representative_objectives)  # type: ignore[arg-type]
                cluster["signals"].append(signal)  # type: ignore[union-attr]
                if candidate_rank > representative_rank:
                    self._record_diagnostic(
                        generation=generation,
                        individual=representative,
                        outcome="peer_replaced",
                        reason="same_generation_peer_cluster",
                        objectives=representative_objectives,  # type: ignore[arg-type]
                    )
                    cluster["representative"] = individual
                    cluster["objectives"] = objectives
                    cluster["key"] = key
                    cluster["signal"] = signal
                    cluster["peer_index"] = cluster_index
                    cluster["diagnostic"] = None
                else:
                    self._record_diagnostic(
                        generation=generation,
                        individual=individual,
                        outcome="rejected",
                        reason="same_generation_peer_cluster",
                        objectives=objectives,
                        correlation=peer_check.get("abs_corr"),
                        matched_index=peer_check.get("matched_index"),
                    )
                break

            if not placed:
                clusters.append(
                    {
                        "representative": individual,
                        "objectives": objectives,
                        "key": key,
                        "signal": signal,
                        "signals": [signal],
                        "peer_index": len(clusters),
                        "diagnostic": None,
                    }
                )

        for cluster in clusters:
            individual = cluster["representative"]
            signal = cluster["signal"]
            key = cluster["key"]
            objectives = cluster["objectives"]
            self.history.append(individual)
            self._history_keys.add(key)
            self._history_signals.append(signal.copy())
            self.admission_objectives[key] = tuple(objectives)
            self._record_diagnostic(
                generation=generation,
                individual=individual,
                outcome="admitted",
                reason="",
                objectives=objectives,
                proxy_minimum=min(objectives[:ARCHIVE_PROXY_OBJECTIVES]),
            )

    def _signal_for(
        self,
        individual: PhenotypicIndividual,
    ) -> tuple[np.ndarray | None, str | None]:
        try:
            if self.signal_provider is not None:
                raw_signal = self.signal_provider(individual.get_phenotype())
            else:
                metadata = getattr(individual, "metadata", {})
                raw_signal = next(
                    (
                        metadata[name]
                        for name in ("signal", "signal_train", "values")
                        if name in metadata
                    ),
                    individual.get_phenotype(),
                )
        except (
            ArithmeticError,
            AttributeError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            return None, f"signal_error:{type(error).__name__}"
        return _normalize_signal(raw_signal)

    def _record_diagnostic(
        self,
        *,
        generation: int,
        individual: object,
        outcome: str,
        reason: str,
        objectives: Sequence[float] = (),
        proxy_minimum: float | None = None,
        correlation: object = None,
        matched_index: object = None,
    ) -> None:
        key = self._expression_key(individual) if isinstance(individual, PhenotypicIndividual) else ""
        expression = (
            str(individual.get_phenotype())
            if isinstance(individual, PhenotypicIndividual)
            else str(individual)
        )
        record: dict[str, object] = {
            "generation": generation,
            "expression": expression,
            "key": key,
            "outcome": outcome,
            "reason": reason,
        }
        if objectives:
            record["objectives"] = [float(value) for value in objectives]
        if proxy_minimum is not None:
            record["proxy_minimum"] = float(proxy_minimum)
        if correlation is not None:
            value = float(correlation)
            record["abs_corr"] = value if math.isfinite(value) else None
        if matched_index is not None:
            record["matched_index"] = matched_index
        self.admission_diagnostics.append(record)

    @staticmethod
    def _rank_key(objectives: Sequence[float]) -> tuple[float, float]:
        return (
            float(min(objectives[:ARCHIVE_PROXY_OBJECTIVES])),
            -float(objectives[ARCHIVE_PROXY_OBJECTIVES]),
        )


class ActiveSetManager(_ActiveSetManagerBase):
    """Maintain active-set history without owning the canonical archive.

    ``ArchiveStep`` is the only component responsible for the persisted and
    returned Pareto front.  This manager reuses the active-set admission and
    promotion policy from the shared implementation, but its per-generation
    processing only updates history and promotion state.  The archive list
    inherited from the common archive base is intentionally not consulted or
    mutated by ``process_evaluated_population``.
    """

    def process_evaluated_population(
        self,
        problem: Problem,
        population_list: Sequence[PhenotypicIndividual],
        generation: int,
    ) -> list[PhenotypicIndividual]:
        """Record active-set history from an already evaluated population."""

        self._validate_problem(problem)
        self._problem = problem
        evaluated = list(population_list)
        valid = self._valid_unique_with_diagnostics(evaluated, problem, generation)
        generation_front = list(non_dominated(iter(valid), problem))
        generation_front_keys = {
            self._expression_key(individual) for individual in generation_front
        }
        for individual in valid:
            key = self._expression_key(individual)
            if key not in generation_front_keys:
                self._record_diagnostic(
                    generation=generation,
                    individual=individual,
                    outcome="not_pareto",
                    reason="dominated_in_generation",
                )

        self._admit_generation(generation_front, problem, generation)
        self.generation_diagnostics.append(
            {
                "generation": generation,
                "population_size": len(evaluated),
                "valid_size": len(valid),
                "front_size": len(generation_front),
                "history_size": len(self.history),
            }
        )
        return generation_front


__all__ = [
    "ACTIVE_SET_FORMAT_IDENTIFIER",
    "ACTIVE_SET_FORMAT_VERSION",
    "ARCHIVE_PROXY_OBJECTIVES",
    "DEFAULT_ACTIVE_CORRELATION_THRESHOLD",
    "DEFAULT_ARCHIVE_CORRELATION_THRESHOLD",
    "DEFAULT_ARCHIVE_QUALITY_THRESHOLD",
    "DEFAULT_FIRST_PROMOTION_TOP_K",
    "DEFAULT_PROMOTION_ADD_K",
    "DEFAULT_PROMOTION_INTERVAL",
    "DEFAULT_PROMOTION_MEAN_GAIN",
    "DEFAULT_PROMOTION_MIN_GAIN",
    "DEFAULT_PROMOTION_REFRESH_TOP_N",
    "FORMAT_IDENTIFIER",
    "FORMAT_VERSION",
    "OBJECTIVES_PER_ARCHIVE",
    "SNAPSHOT_FORMAT_IDENTIFIER",
    "SNAPSHOT_FORMAT_VERSION",
    "SNAPSHOT_MAPPING_REFERENCE",
    "ActiveSetSnapshot",
    "ArchiveSnapshot",
    "ArchiveStep",
    "ActiveSetManager",
    "absolute_pearson_correlation",
    "build_snapshot_document",
    "correlation_rejection",
    "decode_expression",
    "encode_expression",
    "is_correlated_pairwise",
    "load_active_set_snapshot",
    "load_archive",
    "load_snapshot",
    "validate_archive_quality_threshold",
    "validate_correlation_threshold",
    "write_snapshot",
]

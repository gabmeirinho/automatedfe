"""Persisted tables produced by one final-archive evaluation.

The tables in this module are deliberately independent of the fitted model.
They contain the complete feature-level evidence needed by later analysis
steps, while keeping predictions and estimator state process-local.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Final

EVALUATION_DIRECTORY: Final[str] = "evaluation"
FEATURES_FILENAME: Final[str] = "evaluation/features.csv"
METRICS_FILENAME: Final[str] = "evaluation/metrics.json"
IMPORTANCES_FILENAME: Final[str] = "evaluation/importances.csv"
CORRELATIONS_FILENAME: Final[str] = "evaluation/correlations.csv"
TIMINGS_FILENAME: Final[str] = "evaluation/timings.csv"

# Descriptive aliases used by callers that distinguish these from search
# lifecycle tables.
FINAL_FEATURES_FILENAME: Final[str] = FEATURES_FILENAME
FINAL_METRICS_FILENAME: Final[str] = METRICS_FILENAME
FINAL_IMPORTANCES_FILENAME: Final[str] = IMPORTANCES_FILENAME
FINAL_CORRELATIONS_FILENAME: Final[str] = CORRELATIONS_FILENAME
FINAL_TIMINGS_FILENAME: Final[str] = TIMINGS_FILENAME

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "feature_id",
    "feature_label",
    "search_fold_1",
    "search_fold_2",
    "search_fold_3",
)
IMPORTANCE_COLUMNS: Final[tuple[str, ...]] = (
    "feature_id",
    "feature_label",
    "importance_mean",
    "importance_std",
)
CORRELATION_COLUMNS: Final[tuple[str, ...]] = (
    "feature_id",
    "feature_label",
    "other_feature_id",
    "other_feature_label",
    "spearman",
    "training_row_count",
)
TIMING_COLUMNS: Final[tuple[str, ...]] = (
    "feature_id",
    "feature_label",
    "materialization_seconds",
)

FINAL_FEATURE_COLUMNS: Final[tuple[str, ...]] = FEATURE_COLUMNS
FINAL_IMPORTANCE_COLUMNS: Final[tuple[str, ...]] = IMPORTANCE_COLUMNS
FINAL_CORRELATION_COLUMNS: Final[tuple[str, ...]] = CORRELATION_COLUMNS
FINAL_TIMING_COLUMNS: Final[tuple[str, ...]] = TIMING_COLUMNS

EVALUATION_FORMAT: Final[str] = "automatedfe-final-evaluation"
EVALUATION_SCHEMA_VERSION: Final[int] = 1


@dataclass(frozen=True, slots=True)
class FinalEvaluationTables:
    """Complete, model-free tables for one final evaluation."""

    features: tuple[dict[str, object], ...]
    metrics: dict[str, object]
    importances: tuple[dict[str, object], ...]
    correlations: tuple[dict[str, object], ...]
    timings: tuple[dict[str, object], ...]


def _atomic_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return path


def _csv_text(
    rows: Sequence[Mapping[str, object]],
    columns: tuple[str, ...],
) -> str:
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != set(columns):
            raise ValueError(
                "Evaluation table rows must contain exactly: "
                + ", ".join(columns)
            )
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _read_csv(path: Path, columns: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    try:
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != columns:
                raise ValueError(
                    f"CSV header must be: {', '.join(columns)}"
                )
            return tuple(dict(row) for row in reader)
    except FileNotFoundError:
        raise
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"Cannot read evaluation table {path}: {error}") from error


def _finite_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _metric_document(
    metrics: Mapping[str, object],
    *,
    correlation_row_count: int,
    total_materialization_seconds: float,
) -> dict[str, object]:
    if isinstance(correlation_row_count, bool) or correlation_row_count < 0:
        raise ValueError("correlation_row_count must be a non-negative integer")
    document: dict[str, object] = {
        "format": EVALUATION_FORMAT,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "correlation_training_row_count": int(correlation_row_count),
        "total_materialization_seconds": float(total_materialization_seconds),
    }
    for name, value in metrics.items():
        if name in {"model", "predictions", "accuracy"}:
            raise ValueError(f"Final evaluation metrics must not contain {name!r}")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"Final evaluation metric {name!r} must be finite")
        document[str(name)] = converted
    return document


def write_final_evaluation_tables(
    run_dir: str | PathLike[str],
    tables: FinalEvaluationTables,
) -> dict[str, str]:
    """Write complete evaluation tables and return their relative paths."""

    root = Path(run_dir).resolve()
    paths = {
        "features": root / FEATURES_FILENAME,
        "metrics": root / METRICS_FILENAME,
        "importances": root / IMPORTANCES_FILENAME,
        "correlations": root / CORRELATIONS_FILENAME,
        "timings": root / TIMINGS_FILENAME,
    }
    _atomic_text(paths["features"], _csv_text(tables.features, FEATURE_COLUMNS))
    _atomic_text(paths["importances"], _csv_text(tables.importances, IMPORTANCE_COLUMNS))
    _atomic_text(
        paths["correlations"],
        _csv_text(tables.correlations, CORRELATION_COLUMNS),
    )
    _atomic_text(paths["timings"], _csv_text(tables.timings, TIMING_COLUMNS))

    metrics = dict(tables.metrics)
    metrics.pop("format", None)
    metrics.pop("schema_version", None)
    correlation_rows = int(metrics.pop("correlation_training_row_count"))
    total_materialization = float(metrics.pop("total_materialization_seconds"))
    _atomic_text(
        paths["metrics"],
        json.dumps(
            _metric_document(
                metrics,
                correlation_row_count=correlation_rows,
                total_materialization_seconds=total_materialization,
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )
    return {
        name: path.relative_to(root).as_posix() for name, path in paths.items()
    }


def read_final_evaluation_tables(
    run_dir: str | PathLike[str],
    artifacts: Mapping[str, object] | None = None,
) -> FinalEvaluationTables:
    """Read and validate persisted evaluation tables from a run directory."""

    root = Path(run_dir).resolve()
    expected = {
        "features": FEATURES_FILENAME,
        "metrics": METRICS_FILENAME,
        "importances": IMPORTANCES_FILENAME,
        "correlations": CORRELATIONS_FILENAME,
        "timings": TIMINGS_FILENAME,
    }
    if artifacts is not None:
        if set(artifacts) != set(expected):
            raise ValueError("Evaluation artifacts have an unexpected schema")
        for name, relative in expected.items():
            if artifacts[name] != relative:
                raise ValueError(
                    f"Evaluation artifact {name!r} must be {relative!r}"
                )

    features = _read_csv(root / FEATURES_FILENAME, FEATURE_COLUMNS)
    importances = _read_csv(root / IMPORTANCES_FILENAME, IMPORTANCE_COLUMNS)
    correlations = _read_csv(root / CORRELATIONS_FILENAME, CORRELATION_COLUMNS)
    timings = _read_csv(root / TIMINGS_FILENAME, TIMING_COLUMNS)
    try:
        metrics = json.loads((root / METRICS_FILENAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid evaluation metrics: {error}") from error
    if not isinstance(metrics, dict):
        raise ValueError("Evaluation metrics must be a JSON object")
    if metrics.get("format") != EVALUATION_FORMAT:
        raise ValueError("Unknown final evaluation metrics format")
    if metrics.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError("Unsupported final evaluation metrics schema version")
    for name in ("correlation_training_row_count", "total_materialization_seconds"):
        if name not in metrics:
            raise ValueError(f"Evaluation metrics are missing {name!r}")
    if any(name in metrics for name in ("model", "predictions", "accuracy")):
        raise ValueError("Persisted evaluation metrics contain forbidden state")
    return FinalEvaluationTables(
        features=features,
        metrics=metrics,
        importances=importances,
        correlations=correlations,
        timings=timings,
    )


def build_final_evaluation_tables(diagnostics: object) -> FinalEvaluationTables:
    """Convert a diagnostics object into complete persisted table rows.

    The function intentionally consumes the small diagnostics protocol rather
    than a ``RandomForestClassifier``.  This makes it impossible for the
    persistence layer to accidentally serialize an estimator or predictions.
    """

    result_diagnostics = getattr(diagnostics, "diagnostics", diagnostics)
    if result_diagnostics is None:
        raise ValueError("Final evaluation has no diagnostics to persist")
    diagnostics = result_diagnostics
    feature_ids = tuple(str(value) for value in diagnostics.feature_ids)
    labels = tuple(str(value) for value in diagnostics.feature_labels)
    count = len(feature_ids)
    if len(labels) != count:
        raise ValueError("Feature IDs and labels must have equal lengths")

    fold_scores = tuple(diagnostics.search_fold_scores)
    features: list[dict[str, object]] = []
    for index, (feature_id, label) in enumerate(zip(feature_ids, labels)):
        scores = fold_scores[index] if index < len(fold_scores) else ()
        cells = [
            _finite_or_none(scores[position]) if position < len(scores) else None
            for position in range(3)
        ]
        features.append(
            {
                "feature_id": feature_id,
                "feature_label": label,
                "search_fold_1": cells[0],
                "search_fold_2": cells[1],
                "search_fold_3": cells[2],
            }
        )

    means = tuple(float(value) for value in diagnostics.importance_means)
    variations = tuple(float(value) for value in diagnostics.importance_stds)
    if len(means) != count or len(variations) != count:
        raise ValueError("Importance arrays must align with the feature table")
    importances = tuple(
        {
            "feature_id": feature_id,
            "feature_label": label,
            "importance_mean": mean,
            "importance_std": variation,
        }
        for feature_id, label, mean, variation in zip(
            feature_ids, labels, means, variations
        )
    )

    correlations_array = diagnostics.spearman_correlations
    if getattr(correlations_array, "shape", None) != (count, count):
        raise ValueError("Spearman correlations must be a square feature matrix")
    row_count = int(diagnostics.correlation_training_row_count)
    correlations = tuple(
        {
            "feature_id": feature_ids[row],
            "feature_label": labels[row],
            "other_feature_id": feature_ids[column],
            "other_feature_label": labels[column],
            "spearman": _finite_or_none(correlations_array[row, column]),
            "training_row_count": row_count,
        }
        for row in range(count)
        for column in range(count)
    )

    durations = tuple(float(value) for value in diagnostics.materialization_seconds)
    if len(durations) != count:
        raise ValueError("Materialization durations must align with the feature table")
    timings = tuple(
        {
            "feature_id": feature_id,
            "feature_label": label,
            "materialization_seconds": duration,
        }
        for feature_id, label, duration in zip(feature_ids, labels, durations)
    )
    metrics = {
        str(name): float(value) for name, value in diagnostics.metrics.items()
    }
    metrics["correlation_training_row_count"] = row_count
    metrics["total_materialization_seconds"] = float(
        sum(durations)
    )
    return FinalEvaluationTables(
        features=tuple(features),
        metrics=metrics,
        importances=importances,
        correlations=correlations,
        timings=timings,
    )


# Short names keep the table API discoverable alongside the existing
# ``write_candidates_csv``/``read_candidates_csv`` helpers.
build_evaluation_tables = build_final_evaluation_tables
read_evaluation_tables = read_final_evaluation_tables
write_evaluation_tables = write_final_evaluation_tables


__all__ = [
    "CORRELATION_COLUMNS",
    "CORRELATIONS_FILENAME",
    "EVALUATION_DIRECTORY",
    "EVALUATION_FORMAT",
    "EVALUATION_SCHEMA_VERSION",
    "FINAL_CORRELATION_COLUMNS",
    "FINAL_CORRELATIONS_FILENAME",
    "FINAL_FEATURE_COLUMNS",
    "FINAL_FEATURES_FILENAME",
    "FINAL_IMPORTANCE_COLUMNS",
    "FINAL_IMPORTANCES_FILENAME",
    "FINAL_METRICS_FILENAME",
    "FINAL_TIMING_COLUMNS",
    "FINAL_TIMINGS_FILENAME",
    "FEATURE_COLUMNS",
    "FEATURES_FILENAME",
    "FinalEvaluationTables",
    "IMPORTANCE_COLUMNS",
    "IMPORTANCES_FILENAME",
    "METRICS_FILENAME",
    "TIMING_COLUMNS",
    "TIMINGS_FILENAME",
    "build_final_evaluation_tables",
    "build_evaluation_tables",
    "read_evaluation_tables",
    "read_final_evaluation_tables",
    "write_evaluation_tables",
    "write_final_evaluation_tables",
]

"""Materialize GP individuals from the transaction mmap columns.

The preprocessing materializer stores the source transaction columns. This
module is the bridge used before GP evaluation: it takes an individual (or a
``FeatureSpec``), resolves the source columns from those mmaps, and computes
the individual's aggregation with the compiled sliding-window kernel.

Derived features are deliberately not cached. If an output directory is
configured, the selected feature is written to its own mmap file on every
call, so repeated evaluations have simple, predictable behaviour.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ..transaction_materialization import DEFAULT_MMAP_DIR, load_mmapped_columns
from .feature_spec import FeatureSpec, TimeWindow
from .kernels import aggregate

MERCHANT_ID_COLUMN = "merchant_id"
CREATED_AT_COLUMN = "created_at"
FEATURE_MMAP_SUFFIX = ".mmap"

logger = logging.getLogger(__name__)


def _feature_spec(individual: FeatureSpec | Any) -> FeatureSpec:
    """Return a feature specification from a spec or a GP individual."""

    if isinstance(individual, FeatureSpec):
        return individual
    to_spec = getattr(individual, "to_feature_spec", None)
    if to_spec is None or not callable(to_spec):
        raise TypeError(
            "individual must be a FeatureSpec or expose to_feature_spec()"
        )
    spec = to_spec()
    if not isinstance(spec, FeatureSpec):
        raise TypeError("to_feature_spec() must return a FeatureSpec")
    return spec


def _required_columns(spec: FeatureSpec) -> list[str]:
    columns = [MERCHANT_ID_COLUMN]
    if spec.input_column is not None:
        columns.append(spec.input_column)
    if isinstance(spec.window, TimeWindow):
        columns.append(CREATED_AT_COLUMN)
    return list(dict.fromkeys(columns))


def _validate_columns(
    columns: Mapping[str, np.ndarray], required: list[str]
) -> int:
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(
            "Cannot materialize feature; missing column(s): "
            + ", ".join(sorted(missing))
        )

    rows = len(columns[required[0]])
    for column in required:
        values = columns[column]
        if values.ndim != 1:
            raise ValueError(f"Column {column!r} must be one-dimensional")
        if len(values) != rows:
            raise ValueError(
                f"Column {column!r} has {len(values)} rows, expected {rows}"
            )
    return rows


def materialize_feature(
    individual: FeatureSpec | Any,
    columns: Mapping[str, np.ndarray] | str | PathLike[str],
    *,
    output_path: str | PathLike[str] | None = None,
) -> np.ndarray:
    """Compute and optionally persist the aggregation represented by *individual*.

    ``columns`` is normally the mapping returned by
    :func:`automatedfe.transaction_materialization.load_mmapped_columns`, but it may also
    be the directory containing that mapping's manifest. The result has
    one ``float64`` value per transaction row and excludes the current row,
    matching :func:`~automatedfe.features.aggregate`.

    When *output_path* is supplied, the result is copied into a ``float64``
    memory-mapped file. Existing files are overwritten; this function
    intentionally does not implement feature caching.
    """

    if not isinstance(columns, Mapping):
        columns = load_mmapped_columns(Path(columns))

    spec = _feature_spec(individual)
    required = _required_columns(spec)
    rows = _validate_columns(columns, required)

    result = aggregate(
        columns[MERCHANT_ID_COLUMN],
        columns.get(spec.input_column) if spec.input_column is not None else None,
        columns.get(CREATED_AT_COLUMN),
        spec.aggregation,
        spec.window,
    )
    if output_path is None:
        return result

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    mapped = np.memmap(path, dtype=np.float64, mode="w+", shape=(rows,))
    mapped[:] = result
    mapped.flush()
    return mapped


class FeatureMaterializer:
    """Materialize GP individuals from already materialized source columns.

    *columns* may be a mapping of arrays or the directory containing the
    transaction mmap manifest. Source columns are opened once when this
    object is constructed. No derived aggregation is retained between calls.
    Set *output_dir* to persist each derived feature as
    ``<feature-name>.mmap``; otherwise the result is an in-memory NumPy array.
    """

    def __init__(
        self,
        columns: Mapping[str, np.ndarray] | str | PathLike[str] = DEFAULT_MMAP_DIR,
        *,
        output_dir: str | PathLike[str] | None = None,
    ) -> None:
        if isinstance(columns, Mapping):
            self.columns = columns
        else:
            self.columns = load_mmapped_columns(Path(columns))
        self.output_dir = None if output_dir is None else Path(output_dir).resolve()

    def materialize(self, individual: FeatureSpec | Any) -> np.ndarray:
        """Materialize one GP individual, with no derived-feature cache."""

        spec = _feature_spec(individual)
        output_path = None
        if self.output_dir is not None:
            output_path = self.output_dir / f"{spec.name}{FEATURE_MMAP_SUFFIX}"
        if output_path is None:
            logger.info("Materializing feature: %s", spec.name)
        else:
            logger.info("Materializing feature: %s -> %s", spec.name, output_path)
        return materialize_feature(
            spec,
            self.columns,
            output_path=output_path,
        )

    __call__ = materialize

    def materialize_population(
        self,
        individuals: list[FeatureSpec | Any],
    ) -> None:
        """Materialize an already-generated population in its given order."""

        for individual in individuals:
            self.materialize(individual)


def materialize_individual(
    individual: FeatureSpec | Any,
    columns: Mapping[str, np.ndarray] | str | PathLike[str],
    *,
    output_path: str | PathLike[str] | None = None,
) -> np.ndarray:
    """Compatibility-oriented alias for :func:`materialize_feature`."""

    return materialize_feature(individual, columns, output_path=output_path)


materialize_aggregation = materialize_feature


__all__ = [
    "CREATED_AT_COLUMN",
    "FEATURE_MMAP_SUFFIX",
    "MERCHANT_ID_COLUMN",
    "FeatureMaterializer",
    "materialize_aggregation",
    "materialize_feature",
    "materialize_individual",
]

"""Materialize GP individuals from the transaction mmap columns.

The preprocessing materializer stores the source transaction columns. This
module is the bridge used before GP evaluation: it takes an individual (or a
``FeatureSpec``), resolves the source columns from those mmaps, and computes
the individual's aggregation with the compiled sliding-window kernel.

Derived features are cached by :class:`FeatureMaterializer` for the lifetime
of a materialization run. If an output directory is configured, the first
materialization is also written to its own mmap file; later requests for the
same feature reuse the cached array instead of recomputing or overwriting it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ..transaction_materialization import DEFAULT_MMAP_DIR, load_mmapped_columns
from .feature_spec import FeatureSpec, RowWindow, TimeWindow
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
    memory-mapped file. Existing files are overwritten. Caching is provided by
    :class:`FeatureMaterializer`, which owns the cache for a search run.
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


def _materialize_event_feature(
    spec: FeatureSpec,
    columns: Mapping[str, np.ndarray],
    event_merchants: np.ndarray,
    event_timestamps: np.ndarray,
) -> np.ndarray:
    """Calculate a transaction-history feature at each event timestamp."""

    transaction_merchants = np.asarray(columns[MERCHANT_ID_COLUMN])
    transaction_timestamps = np.asarray(columns[CREATED_AT_COLUMN], dtype=np.int64)
    values = None if spec.input_column is None else np.asarray(columns[spec.input_column])
    result = np.full(len(event_timestamps), np.nan, dtype=np.float64)

    order = np.argsort(event_merchants, kind="stable")
    boundaries = np.flatnonzero(
        event_merchants[order][1:] != event_merchants[order][:-1]
    ) + 1
    for event_indices in np.split(order, boundaries):
        merchant = event_merchants[event_indices[0]]
        start = np.searchsorted(transaction_merchants, merchant, side="left")
        stop = np.searchsorted(transaction_merchants, merchant, side="right")
        if start == stop:
            if spec.input_column is None:
                result[event_indices] = 0.0
            continue

        timestamps = transaction_timestamps[start:stop]
        event_times = event_timestamps[event_indices]
        right = np.searchsorted(timestamps, event_times, side="left")
        if isinstance(spec.window, RowWindow):
            left = np.maximum(0, right - spec.window.rows)
        elif isinstance(spec.window, TimeWindow):
            left = np.searchsorted(
                timestamps,
                event_times - spec.window.microseconds,
                side="left",
            )
        else:
            left = np.zeros_like(right)

        counts = right - left
        if spec.input_column is None:
            result[event_indices] = counts
            continue

        merchant_values = np.nan_to_num(values[start:stop])
        if spec.aggregation.value in {"sum", "mean"}:
            prefix = np.concatenate(
                ([0.0], np.cumsum(merchant_values, dtype=np.float64))
            )
            totals = prefix[right] - prefix[left]
            non_empty = counts > 0
            result[event_indices[non_empty]] = totals[non_empty]
            if spec.aggregation.value == "mean":
                result[event_indices[non_empty]] /= counts[non_empty]
        elif spec.aggregation.value == "max":
            for index, window_start, window_stop in zip(event_indices, left, right):
                if window_start < window_stop:
                    result[index] = np.max(
                        merchant_values[window_start:window_stop]
                    )
        else:
            raise ValueError(f"Unsupported aggregation: {spec.aggregation.value}")

    return result


class FeatureMaterializer:
    """Materialize GP individuals from already materialized source columns.

    *columns* may be a mapping of arrays or the directory containing the
    transaction mmap manifest. Source columns are opened once when this
    object is constructed. Derived aggregations are retained in a per-run
    cache, keyed by their :class:`FeatureSpec`. Set *output_dir* to persist
    each derived feature as
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
        self._cache: dict[FeatureSpec, np.ndarray] = {}
        self._event_cache: dict[
            tuple[FeatureSpec, int, int],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}

    def materialize(self, individual: FeatureSpec | Any) -> np.ndarray:
        """Materialize one GP individual, reusing a feature seen in this run."""

        spec = _feature_spec(individual)
        if spec in self._cache:
            logger.info("Reusing cached feature: %s", spec.name)
            return self._cache[spec]

        output_path = None
        if self.output_dir is not None:
            output_path = self.output_dir / f"{spec.name}{FEATURE_MMAP_SUFFIX}"
        if output_path is None:
            logger.info("Materializing feature: %s", spec.name)
        else:
            logger.info("Materializing feature: %s -> %s", spec.name, output_path)
        result = materialize_feature(
            spec,
            self.columns,
            output_path=output_path,
        )
        self._cache[spec] = result
        return result

    __call__ = materialize

    def materialize_population(
        self,
        individuals: list[FeatureSpec | Any],
    ) -> None:
        """Materialize an already-generated population in its given order."""

        for individual in individuals:
            self.materialize(individual)

    def materialize_for_events(
        self,
        individual: FeatureSpec | Any,
        event_merchants: np.ndarray,
        event_timestamps: np.ndarray,
    ) -> np.ndarray:
        """Materialize *individual* and return its value at each event."""

        spec = _feature_spec(individual)
        self.materialize(spec)
        cache_key = (spec, id(event_merchants), id(event_timestamps))
        if cache_key not in self._event_cache:
            self._event_cache[cache_key] = (
                event_merchants,
                event_timestamps,
                _materialize_event_feature(
                    spec,
                    self.columns,
                    np.asarray(event_merchants),
                    np.asarray(event_timestamps, dtype=np.int64),
                ),
            )
        return self._event_cache[cache_key][2]


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

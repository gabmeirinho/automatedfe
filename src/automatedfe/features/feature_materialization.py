"""Materialize primitive features and complete grammar expressions."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ..transaction_materialization import DEFAULT_MMAP_DIR, load_mmapped_columns
from .event_kernels import compute_event_feature
from .feature_types import TxFeature

MERCHANT_ID_COLUMN = "merchant_id"
CREATED_AT_COLUMN = "created_at"
FEATURE_MMAP_SUFFIX = ".mmap"
EVENT_FEATURE_MMAP_SUFFIX = ".events.mmap"
EVENT_FEATURE_METADATA_SUFFIX = ".events.json"

logger = logging.getLogger(__name__)


def _individual_name(individual: TxFeature | Any) -> str:
    if isinstance(individual, TxFeature):
        return individual.name
    return str(individual)


def _primitive_feature(individual: TxFeature | Any) -> TxFeature:
    """Resolve a grammar leaf or descriptor to the canonical feature class."""

    if isinstance(individual, TxFeature):
        return individual

    # CategoryRate is an expression despite exposing to_feature_spec() for
    # compatibility: its value depends on both numerator and denominator.
    from .grammar import CategoryRate, NON_TERMINALS

    if isinstance(individual, CategoryRate) or isinstance(individual, NON_TERMINALS):
        raise TypeError("The expression must be evaluated from its primitive dependencies")

    to_feature = getattr(individual, "to_feature_spec", None)
    if to_feature is None or not callable(to_feature):
        raise TypeError(
            "individual must be a TxFeature or expose to_feature_spec()"
        )
    feature = to_feature()
    if not isinstance(feature, TxFeature):
        raise TypeError("to_feature_spec() must return a TxFeature")
    return feature


def _is_primitive(individual: TxFeature | Any) -> bool:
    """Whether *individual* is a baseline feature worth caching on disk.

    Composed expressions (non-terminals and category rates) are recomputed
    from their cached primitive dependencies instead of being persisted.
    """

    try:
        _primitive_feature(individual)
    except TypeError:
        return False
    return True


def _primitive_dependencies(individual: Any) -> set[TxFeature]:
    from .grammar import collect_features

    return collect_features(individual)


def _cache_stem(name: str) -> str:
    """Return a filesystem-safe cache stem while preserving simple names."""

    if name and not any(character in name for character in "/\\\0"):
        return name
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=16).hexdigest()
    return f"expression_{digest}"


def _category_values(
    feature: TxFeature,
    columns: Mapping[str, np.ndarray],
) -> np.ndarray | None:
    if feature.category_family is None:
        return None
    candidates = (
        feature.category_family,
        f"{feature.category_family}_code",
    )
    for column in candidates:
        if column in columns:
            return np.asarray(columns[column])
    raise ValueError(
        f"Cannot materialize categorical feature {feature.name!r}; "
        f"missing category column for family {feature.category_family!r}"
    )


def _event_columns(
    columns: Mapping[str, np.ndarray],
    *,
    require_amount: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare and validate transaction columns for event feature computation."""

    if MERCHANT_ID_COLUMN not in columns:
        raise ValueError("Cannot materialize feature; missing column(s): merchant_id")
    merchant_ids = np.asarray(columns[MERCHANT_ID_COLUMN])
    rows = len(merchant_ids)
    timestamps = columns.get(CREATED_AT_COLUMN)
    if timestamps is None:
        # Row-window unit tests and callers with synthetic rows need no real
        # clock; a monotonic sequence preserves preceding-row semantics.
        timestamps = np.arange(rows, dtype=np.int64)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if merchant_ids.ndim != 1 or timestamps.ndim != 1:
        raise ValueError("Transaction columns must be one-dimensional")
    if len(timestamps) != rows:
        raise ValueError("merchant_id and created_at must have equal lengths")
    amount = columns.get("amount")
    if amount is None:
        if require_amount:
            raise ValueError("Cannot materialize feature; missing column(s): amount")
        amount = np.zeros(rows, dtype=np.float64)
    amount = np.asarray(amount)
    if amount.ndim != 1 or len(amount) != rows:
        raise ValueError("amount must match merchant_id row count")
    return merchant_ids, timestamps, amount


def _compute_primitive(
    feature: TxFeature,
    columns: Mapping[str, np.ndarray],
    event_merchants: np.ndarray,
    event_timestamps: np.ndarray,
) -> np.ndarray:
    tx_merchants, tx_timestamps, tx_amount = _event_columns(
        columns,
        require_amount=feature.kind not in {"count_total", "count_category"},
    )
    return compute_event_feature(
        tx_merchants,
        tx_timestamps,
        tx_amount,
        _category_values(feature, columns),
        event_merchants,
        event_timestamps,
        kind=feature.kind,
        window_type=feature.window_type,
        window=feature.window,
        category_code=feature.category_code,
    )


def materialize_feature(
    individual: TxFeature | Any,
    columns: Mapping[str, np.ndarray] | str | PathLike[str],
    *,
    output_path: str | PathLike[str] | None = None,
) -> np.ndarray:
    """Materialize one descriptor or complete grammar expression."""

    if not isinstance(columns, Mapping):
        columns = load_mmapped_columns(Path(columns))
    materializer = FeatureMaterializer(columns)
    result = materializer.materialize(individual)
    if output_path is None:
        return result

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    mapped = np.memmap(path, dtype=np.float64, mode="w+", shape=result.shape)
    mapped[:] = result
    mapped.flush()
    return mapped


class FeatureMaterializer:
    """Materialize ``TxFeature`` leaves and complete grammar expressions."""

    def __init__(
        self,
        columns: Mapping[str, np.ndarray] | str | PathLike[str] = DEFAULT_MMAP_DIR,
        *,
        output_dir: str | PathLike[str] | None = None,
        features_dir: str | PathLike[str] | None = None,
    ) -> None:
        if isinstance(columns, Mapping):
            self.columns = columns
        else:
            self.columns = load_mmapped_columns(Path(columns))
        self.output_dir = None if output_dir is None else Path(output_dir).resolve()
        self.features_dir = (
            None if features_dir is None else Path(features_dir).resolve()
        )
        self._cache: dict[str, tuple[np.ndarray, float]] = {}
        self._event_cache: dict[
            tuple[str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray, float]
        ] = {}

    def _transaction_events(self) -> tuple[np.ndarray, np.ndarray]:
        merchants, timestamps, _ = _event_columns(self.columns)
        return merchants, timestamps

    def materialize(self, individual: TxFeature | Any) -> np.ndarray:
        """Materialize one individual over the transaction rows."""

        return self.materialize_with_duration(individual)[0]

    def materialize_with_duration(
        self,
        individual: TxFeature | Any,
    ) -> tuple[np.ndarray, float]:
        """Materialize one individual and return ``(values, duration)``.

        The duration is the monotonic wall-clock time of this top-level
        call. On an in-memory cache hit the previously recorded duration is
        reused unchanged.
        """

        name = _individual_name(individual)
        cached = self._cache.get(name)
        if cached is not None:
            logger.info("Reusing cached feature: %s", name)
            return cached

        started = time.monotonic()
        event_merchants, event_timestamps = self._transaction_events()
        result = self.materialize_for_events_with_duration(
            individual, event_merchants, event_timestamps
        )[0]
        if self.output_dir is not None and _is_primitive(individual):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"{_cache_stem(name)}{FEATURE_MMAP_SUFFIX}"
            mapped = np.memmap(path, dtype=np.float64, mode="w+", shape=result.shape)
            mapped[:] = result
            mapped.flush()
            result = mapped
        duration = time.monotonic() - started
        self._cache[name] = (result, duration)
        return self._cache[name]

    __call__ = materialize

    def materialize_population(self, individuals: list[TxFeature | Any]) -> None:
        """Materialize an already-generated population in its given order."""

        for individual in individuals:
            self.materialize(individual)

    @staticmethod
    def _event_set_checksum(
        event_merchants: np.ndarray,
        event_timestamps: np.ndarray,
    ) -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.ascontiguousarray(event_merchants).view(np.uint8))
        digest.update(np.ascontiguousarray(event_timestamps).view(np.uint8))
        return digest.hexdigest()

    def _load_event_feature(
        self,
        name: str,
        event_merchants: np.ndarray,
        event_timestamps: np.ndarray,
    ) -> tuple[np.ndarray, float] | None:
        """Load a cached feature, returning ``(values, duration)`` or ``None``.

        Metadata written before durations were recorded is treated as a cache
        miss so that the feature is recomputed once and its metadata repaired.
        """

        if self.features_dir is None:
            return None

        checksum = self._event_set_checksum(event_merchants, event_timestamps)
        stem = _cache_stem(name)
        cache_path = self.features_dir / f"{stem}{EVENT_FEATURE_MMAP_SUFFIX}"
        metadata_path = self.features_dir / f"{stem}{EVENT_FEATURE_METADATA_SUFFIX}"
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(metadata, dict):
            return None
        duration = metadata.get("duration")
        if (
            metadata.get("checksum") != checksum
            or metadata.get("rows") != len(event_timestamps)
            or (
                metadata.get("name") is not None
                and metadata.get("name") != name
            )
            or not isinstance(duration, (int, float))
            or not np.isfinite(duration)
            or duration < 0
        ):
            return None
        try:
            cached = np.fromfile(cache_path, dtype=np.float64, count=len(event_timestamps))
        except OSError:
            return None
        if cached.size != len(event_timestamps):
            return None
        logger.info("Reusing cached event feature from disk: %s", name)
        return cached, float(duration)

    def _store_event_feature(
        self,
        name: str,
        event_merchants: np.ndarray,
        event_timestamps: np.ndarray,
        values: np.ndarray,
        duration: float,
    ) -> None:
        if self.features_dir is None:
            return
        self.features_dir.mkdir(parents=True, exist_ok=True)
        checksum = self._event_set_checksum(event_merchants, event_timestamps)
        stem = _cache_stem(name)
        cache_path = self.features_dir / f"{stem}{EVENT_FEATURE_MMAP_SUFFIX}"
        metadata_path = self.features_dir / f"{stem}{EVENT_FEATURE_METADATA_SUFFIX}"
        mapped = np.memmap(cache_path, dtype=np.float64, mode="w+", shape=values.shape)
        mapped[:] = values
        mapped.flush()
        metadata_path.write_text(
            json.dumps(
                {
                    "name": name,
                    "rows": int(len(values)),
                    "checksum": checksum,
                    "duration": float(duration),
                }
            )
        )

    def _materialize_for_events(
        self,
        individual: TxFeature | Any,
        event_merchants: np.ndarray,
        event_timestamps: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Materialize one individual, checking the disk cache for its terminals.

        Only terminal features are cached on disk. Composed expressions are
        recomputed from their cached terminal dependencies.
        """

        try:
            feature = _primitive_feature(individual)
        except TypeError:
            feature = None
        if feature is None:
            started = time.monotonic()
            feature_values = {
                dependency.name: self._materialize_primitive_for_events(
                    dependency, event_merchants, event_timestamps
                )[0]
                for dependency in _primitive_dependencies(individual)
            }
            values = np.asarray(individual.evaluate(feature_values), dtype=np.float64)
            return values, time.monotonic() - started
        return self._materialize_primitive_for_events(
            feature, event_merchants, event_timestamps
        )

    def _materialize_primitive_for_events(
        self,
        feature: TxFeature,
        event_merchants: np.ndarray,
        event_timestamps: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Materialize one terminal feature, reusing the disk cache when possible."""

        name = feature.name
        cache_key = (name, id(event_merchants), id(event_timestamps))
        cached = self._event_cache.get(cache_key)
        if cached is not None:
            return cached[2], cached[3]

        loaded = self._load_event_feature(name, event_merchants, event_timestamps)
        if loaded is not None:
            values, duration = loaded
        else:
            logger.info("Materializing event feature: %s", name)
            started = time.monotonic()
            values = _compute_primitive(
                feature,
                self.columns,
                event_merchants,
                event_timestamps,
            )
            duration = time.monotonic() - started
            self._store_event_feature(
                name, event_merchants, event_timestamps, values, duration
            )
        self._event_cache[cache_key] = (
            event_merchants,
            event_timestamps,
            values,
            duration,
        )
        return values, duration

    def materialize_for_events(
        self,
        individual: TxFeature | Any,
        event_merchants: np.ndarray,
        event_timestamps: np.ndarray,
    ) -> np.ndarray:
        """Materialize an individual at each event row."""

        return self.materialize_for_events_with_duration(
            individual, event_merchants, event_timestamps
        )[0]

    def materialize_for_events_with_duration(
        self,
        individual: TxFeature | Any,
        event_merchants: np.ndarray,
        event_timestamps: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Materialize an individual at each event row.

        Returns ``(values, duration)``. Only terminal features check the disk
        cache; composed expressions are evaluated from their cached terminals.
        """

        event_merchants = np.asarray(event_merchants, dtype=np.int64)
        event_timestamps = np.asarray(event_timestamps, dtype=np.int64)
        name = _individual_name(individual)
        cache_key = (name, id(event_merchants), id(event_timestamps))
        cached = self._event_cache.get(cache_key)
        if cached is not None:
            return cached[2], cached[3]

        values, duration = self._materialize_for_events(
            individual, event_merchants, event_timestamps
        )
        self._event_cache[cache_key] = (
            event_merchants,
            event_timestamps,
            values,
            duration,
        )
        return values, duration


def materialize_individual(
    individual: TxFeature | Any,
    columns: Mapping[str, np.ndarray] | str | PathLike[str],
    *,
    output_path: str | PathLike[str] | None = None,
) -> np.ndarray:
    """Compatibility alias for :func:`materialize_feature`."""

    return materialize_feature(individual, columns, output_path=output_path)


materialize_aggregation = materialize_feature


__all__ = [
    "CREATED_AT_COLUMN",
    "EVENT_FEATURE_METADATA_SUFFIX",
    "EVENT_FEATURE_MMAP_SUFFIX",
    "FEATURE_MMAP_SUFFIX",
    "MERCHANT_ID_COLUMN",
    "FeatureMaterializer",
    "materialize_aggregation",
    "materialize_feature",
    "materialize_individual",
]

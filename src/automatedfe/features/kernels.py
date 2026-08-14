"""Compiled sliding-window aggregation kernels.

The transaction rows are sorted by ``merchant_id, created_at``, so every row
of one merchant is contiguous and timestamps never decrease within a group.
Each kernel consumes those arrays and writes, for every row ``i``, the value of
an aggregation over the merchant's *preceding* rows that fall inside a window:
the last ``N`` rows, the last ``T`` microseconds, or the entire history up to
``i``. The row itself is always excluded from its own window.

The implementations mirror the production kernels in ``gp-benchmarks``:
row windows use bounded ring buffers, time windows use two-pointer scans, and
maxima use a dynamically growing monotonic deque.  Nothing is recomputed from
scratch per row and row-window state is bounded by the requested window size.

Empty windows (the first row of a merchant, or no row within a time window)
produce ``0`` for ``count`` and ``NaN`` for every value aggregation. Input
values must be free of ``NaN`` (the materialization pipeline zero-fills nulls
before this step).
"""

from __future__ import annotations

import math

import numba
import numpy as np

from .feature_spec import RowWindow, TimeWindow, TotalHistoryWindow, Window

COUNT = 0
SUM = 1
MEAN = 2
MAX = 3

ROW_WINDOW = 0
TIME_WINDOW = 1
TOTAL_HISTORY = 2

_AGGREGATIONS: dict[str, int] = {
    "count": COUNT,
    "sum": SUM,
    "mean": MEAN,
    "max": MAX,
}

_WINDOW_MODES: dict[str, int] = {
    "rows": ROW_WINDOW,
    "time": TIME_WINDOW,
    "total": TOTAL_HISTORY,
}


@numba.njit(cache=True, nogil=True)
def _count_row_kernel(
    merchant_id: np.ndarray,
    result: np.ndarray,
    window_span: int,
) -> None:
    """Write counts over the preceding ``window_span`` merchant rows."""

    merchant_start = 0
    for i in range(merchant_id.shape[0]):
        if i == 0 or merchant_id[i] != merchant_id[i - 1]:
            merchant_start = i
        result[i] = float(i - max(merchant_start, i - window_span))


@numba.njit(cache=True, nogil=True)
def _count_time_kernel(
    merchant_id: np.ndarray,
    created_at: np.ndarray,
    result: np.ndarray,
    window_span: int,
) -> None:
    """Write counts over the preceding timestamp window."""

    left = 0
    for i in range(merchant_id.shape[0]):
        if i == 0 or merchant_id[i] != merchant_id[i - 1]:
            left = i
        min_timestamp = created_at[i] - window_span
        while left < i and created_at[left] < min_timestamp:
            left += 1
        result[i] = float(i - left)


@numba.njit(cache=True, nogil=True)
def _count_total_kernel(merchant_id: np.ndarray, result: np.ndarray) -> None:
    """Write counts over all preceding merchant rows."""

    merchant_start = 0
    for i in range(merchant_id.shape[0]):
        if i == 0 or merchant_id[i] != merchant_id[i - 1]:
            merchant_start = i
        result[i] = float(i - merchant_start)


@numba.njit(cache=True, nogil=True)
def _sum_row_kernel(
    merchant_id: np.ndarray,
    values: np.ndarray,
    result: np.ndarray,
    window_span: int,
) -> None:
    """Write sums from a bounded row-window ring buffer."""

    ring = np.empty(window_span, dtype=np.float64)
    ring_start = 0
    ring_count = 0
    total = 0.0
    for i in range(merchant_id.shape[0]):
        if i == 0 or merchant_id[i] != merchant_id[i - 1]:
            ring_start = 0
            ring_count = 0
            total = 0.0

        result[i] = total if ring_count > 0 else math.nan

        value = values[i]
        if ring_count < window_span:
            ring[(ring_start + ring_count) % window_span] = value
            ring_count += 1
        else:
            total -= ring[ring_start]
            ring[ring_start] = value
            ring_start = (ring_start + 1) % window_span
        total += value


@numba.njit(cache=True, nogil=True)
def _sum_time_kernel(
    merchant_id: np.ndarray,
    values: np.ndarray,
    created_at: np.ndarray,
    result: np.ndarray,
    window_span: int,
) -> None:
    """Write sums from a two-pointer timestamp window."""

    left = 0
    count = 0
    total = 0.0
    for i in range(merchant_id.shape[0]):
        if i == 0 or merchant_id[i] != merchant_id[i - 1]:
            left = i
            count = 0
            total = 0.0

        min_timestamp = created_at[i] - window_span
        while left < i and created_at[left] < min_timestamp:
            total -= values[left]
            count -= 1
            left += 1

        result[i] = total if count > 0 else math.nan
        total += values[i]
        count += 1


@numba.njit(cache=True, nogil=True)
def _sum_total_kernel(
    merchant_id: np.ndarray,
    values: np.ndarray,
    result: np.ndarray,
) -> None:
    """Write sums over all preceding merchant rows."""

    count = 0
    total = 0.0
    for i in range(merchant_id.shape[0]):
        if i == 0 or merchant_id[i] != merchant_id[i - 1]:
            count = 0
            total = 0.0
        result[i] = total if count > 0 else math.nan
        total += values[i]
        count += 1


@numba.njit(cache=True, nogil=True)
def _mean_row_kernel(
    merchant_id: np.ndarray,
    values: np.ndarray,
    result: np.ndarray,
    window_span: int,
) -> None:
    """Write means from a bounded row-window ring buffer."""

    ring = np.empty(window_span, dtype=np.float64)
    ring_start = 0
    ring_count = 0
    total = 0.0
    for i in range(merchant_id.shape[0]):
        if i == 0 or merchant_id[i] != merchant_id[i - 1]:
            ring_start = 0
            ring_count = 0
            total = 0.0

        result[i] = total / ring_count if ring_count > 0 else math.nan

        value = values[i]
        if ring_count < window_span:
            ring[(ring_start + ring_count) % window_span] = value
            ring_count += 1
        else:
            total -= ring[ring_start]
            ring[ring_start] = value
            ring_start = (ring_start + 1) % window_span
        total += value


@numba.njit(cache=True, nogil=True)
def _mean_time_kernel(
    merchant_id: np.ndarray,
    values: np.ndarray,
    created_at: np.ndarray,
    result: np.ndarray,
    window_span: int,
) -> None:
    """Write means from a two-pointer timestamp window."""

    left = 0
    count = 0
    total = 0.0
    for i in range(merchant_id.shape[0]):
        if i == 0 or merchant_id[i] != merchant_id[i - 1]:
            left = i
            count = 0
            total = 0.0

        min_timestamp = created_at[i] - window_span
        while left < i and created_at[left] < min_timestamp:
            total -= values[left]
            count -= 1
            left += 1

        result[i] = total / count if count > 0 else math.nan
        total += values[i]
        count += 1


@numba.njit(cache=True, nogil=True)
def _mean_total_kernel(
    merchant_id: np.ndarray,
    values: np.ndarray,
    result: np.ndarray,
) -> None:
    """Write means over all preceding merchant rows."""

    count = 0
    total = 0.0
    for i in range(merchant_id.shape[0]):
        if i == 0 or merchant_id[i] != merchant_id[i - 1]:
            count = 0
            total = 0.0
        result[i] = total / count if count > 0 else math.nan
        total += values[i]
        count += 1


@numba.njit(inline="always")
def _append_maximum(
    deque_indices: np.ndarray,
    head: int,
    tail: int,
    capacity: int,
    values: np.ndarray,
    index: int,
) -> tuple[np.ndarray, int, int, int]:
    """Append one row to a compacting, dynamically growing monotonic deque."""

    if tail >= capacity:
        if head > 0:
            size = tail - head
            for j in range(size):
                deque_indices[j] = deque_indices[head + j]
            head = 0
            tail = size
        else:
            new_capacity = capacity * 2
            new_deque = np.empty(new_capacity, dtype=np.int64)
            for j in range(tail):
                new_deque[j] = deque_indices[j]
            deque_indices = new_deque
            capacity = new_capacity

    value = values[index]
    while tail > head and values[deque_indices[tail - 1]] <= value:
        tail -= 1
    deque_indices[tail] = index
    return deque_indices, head, tail + 1, capacity


@numba.njit(cache=True, nogil=True)
def _max_row_kernel(
    merchant_id: np.ndarray,
    values: np.ndarray,
    result: np.ndarray,
    window_span: int,
) -> None:
    """Write row-window maxima using a monotonic deque."""

    capacity = 1024
    deque_indices = np.empty(capacity, dtype=np.int64)
    head = 0
    tail = 0
    merchant_start = 0
    for i in range(merchant_id.shape[0]):
        if i == 0 or merchant_id[i] != merchant_id[i - 1]:
            merchant_start = i
            head = 0
            tail = 0

        left = max(merchant_start, i - window_span)
        while head < tail and deque_indices[head] < left:
            head += 1
        result[i] = values[deque_indices[head]] if head < tail else math.nan
        deque_indices, head, tail, capacity = _append_maximum(
            deque_indices, head, tail, capacity, values, i
        )


@numba.njit(cache=True, nogil=True)
def _max_time_kernel(
    merchant_id: np.ndarray,
    values: np.ndarray,
    created_at: np.ndarray,
    result: np.ndarray,
    window_span: int,
) -> None:
    """Write timestamp-window maxima using a monotonic deque."""

    capacity = 1024
    deque_indices = np.empty(capacity, dtype=np.int64)
    head = 0
    tail = 0
    left = 0
    for i in range(merchant_id.shape[0]):
        if i == 0 or merchant_id[i] != merchant_id[i - 1]:
            left = i
            head = 0
            tail = 0

        min_timestamp = created_at[i] - window_span
        while left < i and created_at[left] < min_timestamp:
            left += 1
        while head < tail and deque_indices[head] < left:
            head += 1

        result[i] = values[deque_indices[head]] if head < tail else math.nan
        deque_indices, head, tail, capacity = _append_maximum(
            deque_indices, head, tail, capacity, values, i
        )


@numba.njit(cache=True, nogil=True)
def _max_total_kernel(
    merchant_id: np.ndarray,
    values: np.ndarray,
    result: np.ndarray,
) -> None:
    """Write maxima over all preceding merchant rows."""

    capacity = 1024
    deque_indices = np.empty(capacity, dtype=np.int64)
    head = 0
    tail = 0
    for i in range(merchant_id.shape[0]):
        if i == 0 or merchant_id[i] != merchant_id[i - 1]:
            head = 0
            tail = 0
        result[i] = values[deque_indices[head]] if head < tail else math.nan
        deque_indices, head, tail, capacity = _append_maximum(
            deque_indices, head, tail, capacity, values, i
        )


def _resolve_aggregation(aggregation: str) -> int:
    try:
        return _AGGREGATIONS[aggregation]
    except KeyError:
        choices = ", ".join(sorted(_AGGREGATIONS))
        raise ValueError(f"Unknown aggregation {aggregation!r}; expected one of: {choices}") from None


def _resolve_window_mode(window_mode: str) -> int:
    try:
        return _WINDOW_MODES[window_mode]
    except KeyError:
        choices = ", ".join(sorted(_WINDOW_MODES))
        raise ValueError(f"Unknown window_mode {window_mode!r}; expected one of: {choices}") from None


def _prepare(merchant_id, values, timestamps, *, needs_values: bool, window_mode: int) -> tuple:
    merchant_id = np.ascontiguousarray(merchant_id)
    if merchant_id.ndim != 1 or merchant_id.dtype.kind not in "iu":
        raise TypeError("merchant_id must be a 1-D integer array")
    merchant_id = merchant_id.astype(np.int64)

    if needs_values:
        if values is None:
            raise ValueError("aggregation requires the 'values' array")
        values = np.ascontiguousarray(values, dtype=np.float64)
        if values.ndim != 1:
            raise TypeError("values must be a 1-D array")
        if values.shape[0] != merchant_id.shape[0]:
            raise ValueError(
                f"values has {values.shape[0]} rows, expected {merchant_id.shape[0]}"
            )
    else:
        values = np.empty(0, dtype=np.float64)

    if window_mode == TIME_WINDOW:
        if timestamps is None:
            raise ValueError("time windows require the 'timestamps' array")
        timestamps = np.ascontiguousarray(timestamps, dtype=np.int64)
        if timestamps.ndim != 1:
            raise TypeError("timestamps must be a 1-D array")
        if timestamps.shape[0] != merchant_id.shape[0]:
            raise ValueError(
                f"timestamps has {timestamps.shape[0]} rows, expected {merchant_id.shape[0]}"
            )
    else:
        timestamps = np.empty(0, dtype=np.int64)

    return merchant_id, values, timestamps


def sliding_window(
    merchant_id: np.ndarray,
    values: np.ndarray | None,
    timestamps: np.ndarray | None,
    aggregation: str = "count",
    window_mode: str = "rows",
    window_span: int = 0,
) -> np.ndarray:
    """Compute a sliding-window aggregation for every row.

    ``aggregation`` is one of ``"count"``, ``"sum"``, ``"mean"``, or
    ``"max"``. ``count`` ignores *values*; the rest require it. *timestamps*
    (int64 microseconds)
    is only required for ``window_mode="time"``. *window_span* is a row count
    for ``"rows"`` and a microsecond span for ``"time"``; it is ignored for
    ``"total"``.

    Returns a ``float64`` array with one entry per row. Empty windows yield
    ``0`` for ``count`` and ``NaN`` otherwise.
    """

    aggregation_code = _resolve_aggregation(aggregation)
    mode_code = _resolve_window_mode(window_mode)
    needs_values = aggregation_code != COUNT

    merchant_id, values, timestamps = _prepare(
        merchant_id, values, timestamps, needs_values=needs_values, window_mode=mode_code
    )

    if window_span <= 0 and mode_code != TOTAL_HISTORY:
        raise ValueError(
            f"window_span must be positive for window_mode={window_mode!r}, got {window_span}"
        )

    result = np.empty(merchant_id.shape[0], dtype=np.float64)
    if mode_code == ROW_WINDOW:
        if aggregation_code == COUNT:
            _count_row_kernel(merchant_id, result, window_span)
        elif aggregation_code == SUM:
            _sum_row_kernel(merchant_id, values, result, window_span)
        elif aggregation_code == MEAN:
            _mean_row_kernel(merchant_id, values, result, window_span)
        elif aggregation_code == MAX:
            _max_row_kernel(merchant_id, values, result, window_span)
        else:
            raise RuntimeError(f"Unsupported aggregation code: {aggregation_code}")
    elif mode_code == TIME_WINDOW:
        if aggregation_code == COUNT:
            _count_time_kernel(merchant_id, timestamps, result, window_span)
        elif aggregation_code == SUM:
            _sum_time_kernel(
                merchant_id, values, timestamps, result, window_span
            )
        elif aggregation_code == MEAN:
            _mean_time_kernel(
                merchant_id, values, timestamps, result, window_span
            )
        elif aggregation_code == MAX:
            _max_time_kernel(
                merchant_id, values, timestamps, result, window_span
            )
        else:
            raise RuntimeError(f"Unsupported aggregation code: {aggregation_code}")
    elif mode_code == TOTAL_HISTORY:
        if aggregation_code == COUNT:
            _count_total_kernel(merchant_id, result)
        elif aggregation_code == SUM:
            _sum_total_kernel(merchant_id, values, result)
        elif aggregation_code == MEAN:
            _mean_total_kernel(merchant_id, values, result)
        elif aggregation_code == MAX:
            _max_total_kernel(merchant_id, values, result)
        else:
            raise RuntimeError(f"Unsupported aggregation code: {aggregation_code}")
    else:
        raise RuntimeError(f"Unsupported window mode code: {mode_code}")
    return result


def aggregate(
    merchant_id: np.ndarray,
    values: np.ndarray | None,
    timestamps: np.ndarray | None,
    aggregation: str,
    window: Window,
) -> np.ndarray:
    """Compute *aggregation* over *window* for every transaction row.

    Convenience wrapper around :func:`sliding_window` accepting the
    window and aggregation value objects used by :class:`TxFeature`.
    ``"count"`` ignores *values*; the other aggregations require the ``"amount"``
    column in *values*. *timestamps* is only used for
    :class:`~automatedfe.features.TimeWindow` windows.
    """

    if isinstance(window, RowWindow):
        window_mode, window_span = "rows", window.rows
    elif isinstance(window, TimeWindow):
        window_mode, window_span = "time", window.microseconds
    elif isinstance(window, TotalHistoryWindow):
        window_mode, window_span = "total", 0
    else:
        raise TypeError(f"Unsupported window type: {type(window).__name__}")

    return sliding_window(
        merchant_id,
        values,
        timestamps,
        aggregation=aggregation,
        window_mode=window_mode,
        window_span=window_span,
    )


__all__ = [
    "COUNT",
    "SUM",
    "MEAN",
    "MAX",
    "ROW_WINDOW",
    "TIME_WINDOW",
    "TOTAL_HISTORY",
    "aggregate",
    "sliding_window",
]

"""Compiled sliding-window aggregation kernels.

The transaction rows are sorted by ``merchant_id, created_at``, so every row
of one merchant is contiguous and timestamps never decrease within a group.
Each kernel consumes those arrays and writes, for every row ``i``, the value of
an aggregation over the merchant's *preceding* rows that fall inside a window:
the last ``N`` rows, the last ``T`` microseconds, or the entire history up to
``i``. The row itself is always excluded from its own window.

Every kernel runs in linear time with a two-pointer scan: when ``i`` advances,
the newly entered row is folded into running state and the row that left the
window is subtracted back out (``sum``/``mean``) or popped off a monotonic
deque (``max``). Nothing is recomputed from scratch per row.

Empty windows (the first row of a merchant, or no row within a time window)
produce ``0`` for ``count`` and ``NaN`` for every value aggregation. Input
values must be free of ``NaN`` (the materialization pipeline zero-fills nulls
before this step).
"""

from __future__ import annotations

import math

import numba
import numpy as np

from .feature_spec import Aggregation, RowWindow, TimeWindow, TotalHistoryWindow, Window

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


@numba.njit(inline="always")
def _window_left(
    merchant_id: np.ndarray,
    created_at: np.ndarray,
    i: int,
    left: int,
    window_mode: int,
    window_span: int,
) -> int:
    """Return the first row included in the window for row *i*."""

    if i > 0 and merchant_id[i] != merchant_id[i - 1]:
        return i

    if window_mode == ROW_WINDOW:
        while i - left > window_span:
            left += 1
    elif window_mode == TIME_WINDOW:
        while left < i and created_at[i] - created_at[left] > window_span:
            left += 1
    return left


@numba.njit(cache=True, nogil=True)
def _count_kernel(
    merchant_id: np.ndarray,
    created_at: np.ndarray,
    result: np.ndarray,
    window_mode: int,
    window_span: int,
) -> None:
    """Write trailing transaction counts into *result*."""

    left = 0
    for i in range(merchant_id.shape[0]):
        left = _window_left(
            merchant_id, created_at, i, left, window_mode, window_span
        )
        result[i] = float(i - left)


@numba.njit(cache=True, nogil=True)
def _sum_kernel(
    merchant_id: np.ndarray,
    values: np.ndarray,
    created_at: np.ndarray,
    result: np.ndarray,
    window_mode: int,
    window_span: int,
) -> None:
    """Write trailing sums into *result*."""

    left = 0
    total = 0.0

    for i in range(merchant_id.shape[0]):
        new_left = _window_left(
            merchant_id, created_at, i, left, window_mode, window_span
        )
        new_merchant = i > 0 and merchant_id[i] != merchant_id[i - 1]
        if new_merchant:
            total = 0.0
        else:
            while left < new_left:
                total -= values[left]
                left += 1
        left = new_left

        count = i - left
        result[i] = total if count > 0 else math.nan
        total += values[i]


@numba.njit(cache=True, nogil=True)
def _mean_kernel(
    merchant_id: np.ndarray,
    values: np.ndarray,
    created_at: np.ndarray,
    result: np.ndarray,
    window_mode: int,
    window_span: int,
) -> None:
    """Write trailing means into *result*."""

    left = 0
    total = 0.0

    for i in range(merchant_id.shape[0]):
        new_left = _window_left(
            merchant_id, created_at, i, left, window_mode, window_span
        )
        new_merchant = i > 0 and merchant_id[i] != merchant_id[i - 1]
        if new_merchant:
            total = 0.0
        else:
            while left < new_left:
                total -= values[left]
                left += 1
        left = new_left

        count = i - left
        result[i] = total / count if count > 0 else math.nan
        total += values[i]


@numba.njit(cache=True, nogil=True)
def _max_kernel(
    merchant_id: np.ndarray,
    values: np.ndarray,
    created_at: np.ndarray,
    result: np.ndarray,
    window_mode: int,
    window_span: int,
) -> None:
    """Write trailing maxima into *result* using a monotonic deque."""

    n = merchant_id.shape[0]
    deque_indices = np.empty(n, dtype=np.int64)
    deque_head = 0
    deque_tail = 0
    left = 0

    for i in range(n):
        new_left = _window_left(
            merchant_id, created_at, i, left, window_mode, window_span
        )
        new_merchant = i > 0 and merchant_id[i] != merchant_id[i - 1]
        if new_merchant:
            left = i
            deque_head = 0
            deque_tail = 0
        else:
            while left < new_left:
                if deque_head < deque_tail and deque_indices[deque_head] == left:
                    deque_head += 1
                left += 1

        left = new_left
        count = i - left
        result[i] = values[deque_indices[deque_head]] if count > 0 else math.nan

        value = values[i]
        while deque_tail > deque_head and values[deque_indices[deque_tail - 1]] <= value:
            deque_tail -= 1
        deque_indices[deque_tail] = i
        deque_tail += 1


def _resolve_aggregation(aggregation: str | Aggregation) -> int:
    if isinstance(aggregation, Aggregation):
        aggregation = aggregation.value
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
    aggregation: str | Aggregation = Aggregation.COUNT,
    window_mode: str = "rows",
    window_span: int = 0,
) -> np.ndarray:
    """Compute a sliding-window aggregation for every row.

    ``aggregation`` is one of ``"count"``, ``"sum"``, ``"mean"``, or
    ``"max"`` (or an :class:`~automatedfe.features.Aggregation`). ``count``
    ignores *values*; the rest require it. *timestamps* (int64 microseconds)
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
    if aggregation_code == COUNT:
        _count_kernel(merchant_id, timestamps, result, mode_code, window_span)
    elif aggregation_code == SUM:
        _sum_kernel(merchant_id, values, timestamps, result, mode_code, window_span)
    elif aggregation_code == MEAN:
        _mean_kernel(merchant_id, values, timestamps, result, mode_code, window_span)
    elif aggregation_code == MAX:
        _max_kernel(merchant_id, values, timestamps, result, mode_code, window_span)
    else:
        raise RuntimeError(f"Unsupported aggregation code: {aggregation_code}")
    return result


def aggregate(
    merchant_id: np.ndarray,
    values: np.ndarray | None,
    timestamps: np.ndarray | None,
    aggregation: Aggregation,
    window: Window,
) -> np.ndarray:
    """Compute *aggregation* over *window* for every transaction row.

    Convenience wrapper around :func:`sliding_window` accepting the
    :class:`~automatedfe.features.FeatureSpec` window and aggregation types.
    ``Aggregation.COUNT`` ignores *values*; the other aggregations require the
    ``"amount"`` column in *values*. *timestamps* is only used for
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

"""Event-level kernels for complete transaction features.

The transaction materializer stores rows ordered by merchant and timestamp,
while the event dataset is usually ordered chronologically.  The public
wrapper below groups events by merchant before calling the compiled kernel and
restores their original order afterwards.
"""

from __future__ import annotations

import numba
import numpy as np

from .feature_schema import DAY_MICROSECONDS, TOTAL_TIME_WINDOW

MEAN = 0
MAX = 1
COUNT_TOTAL = 2
COUNT_CATEGORY = 3
AVG_DAILY_COUNT = 4
AVG_DAILY_COUNT_CATEGORY = 5
AVG_DAILY_AMOUNT = 6
AVG_DAILY_AMOUNT_CATEGORY = 7
AVG_DAILY_TOTAL_AMOUNT = 8
TOTAL_AMOUNT = 9
STD = 10

ROW_WINDOW = 0
TIME_WINDOW = 1
DAY_WINDOW = 2

_KIND_CODES = {
    "mean": MEAN,
    "max": MAX,
    "sum": TOTAL_AMOUNT,
    "count_total": COUNT_TOTAL,
    "count_category": COUNT_CATEGORY,
    "avg_daily_count": AVG_DAILY_COUNT,
    "avg_daily_count_category": AVG_DAILY_COUNT_CATEGORY,
    "avg_daily_amount": AVG_DAILY_AMOUNT,
    "avg_daily_amount_category": AVG_DAILY_AMOUNT_CATEGORY,
    "avg_daily_total_amount": AVG_DAILY_TOTAL_AMOUNT,
    "total_amount": TOTAL_AMOUNT,
    "std": STD,
}

_WINDOW_CODES = {"row": ROW_WINDOW, "time": TIME_WINDOW, "days": DAY_WINDOW}


@numba.njit(cache=True, nogil=True)
def _compute_grouped(
    tx_ids: np.ndarray,
    tx_timestamps: np.ndarray,
    tx_amount: np.ndarray,
    tx_category: np.ndarray,
    event_ids: np.ndarray,
    event_timestamps: np.ndarray,
    output: np.ndarray,
    kind: int,
    window_type: int,
    window: int,
    category_code: int,
) -> None:
    """Compute one primitive feature for merchant-grouped events."""

    event_start = 0
    n_events = event_ids.shape[0]
    n_transactions = tx_ids.shape[0]
    transaction_cursor = 0

    while event_start < n_events:
        merchant_id = event_ids[event_start]
        event_stop = event_start + 1
        while event_stop < n_events and event_ids[event_stop] == merchant_id:
            event_stop += 1

        # Transaction rows and grouped event rows are both ordered by merchant.
        # Advance the merchant cursor once instead of searching from the start
        # for every event group.
        while (
            transaction_cursor < n_transactions
            and tx_ids[transaction_cursor] < merchant_id
        ):
            transaction_cursor += 1
        tx_start = transaction_cursor
        while (
            transaction_cursor < n_transactions
            and tx_ids[transaction_cursor] == merchant_id
        ):
            transaction_cursor += 1
        tx_stop = transaction_cursor

        group_size = tx_stop - tx_start
        prefix_sum = np.empty(group_size + 1, dtype=np.float64)
        prefix_square = np.empty(group_size + 1, dtype=np.float64)
        prefix_category = np.empty(group_size + 1, dtype=np.int64)
        prefix_sum[0] = 0.0
        prefix_square[0] = 0.0
        prefix_category[0] = 0
        for offset in range(group_size):
            value = tx_amount[tx_start + offset]
            prefix_sum[offset + 1] = prefix_sum[offset] + value
            prefix_square[offset + 1] = prefix_square[offset] + value * value
            prefix_category[offset + 1] = (
                prefix_category[offset]
                + (1 if tx_category[tx_start + offset] == category_code else 0)
            )

        # Event timestamps are ordered within each merchant group, so these
        # cursors only move forward as the event window moves forward.
        right_cursor = tx_start
        left_cursor = tx_start
        for event_index in range(event_start, event_stop):
            event_time = event_timestamps[event_index]
            while (
                right_cursor < tx_stop
                and tx_timestamps[right_cursor] < event_time
            ):
                right_cursor += 1
            right = right_cursor - tx_start

            if window_type == ROW_WINDOW:
                left = max(0, right - window)
            elif window_type == TIME_WINDOW:
                if window == TOTAL_TIME_WINDOW:
                    left = 0
                else:
                    window_start = event_time - window
                    while (
                        left_cursor < tx_stop
                        and tx_timestamps[left_cursor] < window_start
                    ):
                        left_cursor += 1
                    left = left_cursor - tx_start
            else:
                event_day = event_time // DAY_MICROSECONDS
                window_start_day = event_day - window + 1
                while (
                    left_cursor < tx_stop
                    and tx_timestamps[left_cursor] // DAY_MICROSECONDS
                    < window_start_day
                ):
                    left_cursor += 1
                left = left_cursor - tx_start

            count = right - left
            total = prefix_sum[right] - prefix_sum[left]
            category_count = prefix_category[right] - prefix_category[left]

            if kind == MEAN:
                output[event_index] = total / count if count else np.nan
            elif kind == MAX:
                if count == 0:
                    output[event_index] = np.nan
                else:
                    maximum = tx_amount[tx_start + left]
                    for offset in range(left + 1, right):
                        maximum = max(maximum, tx_amount[tx_start + offset])
                    output[event_index] = maximum
            elif kind == COUNT_TOTAL:
                output[event_index] = count
            elif kind == COUNT_CATEGORY:
                output[event_index] = category_count
            elif kind == AVG_DAILY_COUNT:
                output[event_index] = count / window
            elif kind == AVG_DAILY_COUNT_CATEGORY:
                output[event_index] = category_count / window
            elif kind == AVG_DAILY_AMOUNT:
                output[event_index] = total / window
            elif kind == AVG_DAILY_AMOUNT_CATEGORY:
                category_total = 0.0
                for offset in range(left, right):
                    absolute = tx_start + offset
                    if tx_category[absolute] == category_code:
                        category_total += tx_amount[absolute]
                output[event_index] = category_total / window
            elif kind == AVG_DAILY_TOTAL_AMOUNT:
                output[event_index] = total / window
            elif kind == TOTAL_AMOUNT:
                output[event_index] = total
            elif kind == STD:
                if count == 0:
                    output[event_index] = np.nan
                else:
                    mean = total / count
                    variance = (
                        prefix_square[right] - prefix_square[left]
                    ) / count - mean * mean
                    output[event_index] = np.sqrt(max(variance, 0.0))

        event_start = event_stop


def compute_event_feature(
    tx_ids: np.ndarray,
    tx_timestamps: np.ndarray,
    tx_amount: np.ndarray,
    tx_category: np.ndarray | None,
    event_ids: np.ndarray,
    event_timestamps: np.ndarray,
    *,
    kind: str,
    window_type: str,
    window: int,
    category_code: int | None = None,
) -> np.ndarray:
    """Compute one primitive transaction feature at arbitrary event rows."""

    try:
        kind_code = _KIND_CODES[kind]
    except KeyError:
        raise ValueError(f"Unsupported transaction feature kind: {kind}") from None
    try:
        window_code = _WINDOW_CODES[window_type]
    except KeyError:
        raise ValueError(f"Unsupported transaction window type: {window_type}") from None

    tx_ids = np.ascontiguousarray(tx_ids, dtype=np.int64)
    tx_timestamps = np.ascontiguousarray(tx_timestamps, dtype=np.int64)
    tx_amount = np.ascontiguousarray(tx_amount, dtype=np.float64)
    event_ids = np.ascontiguousarray(event_ids, dtype=np.int64)
    event_timestamps = np.ascontiguousarray(event_timestamps, dtype=np.int64)
    if tx_ids.ndim != 1 or tx_timestamps.ndim != 1 or tx_amount.ndim != 1:
        raise ValueError("Transaction columns must be one-dimensional")
    if not (len(tx_ids) == len(tx_timestamps) == len(tx_amount)):
        raise ValueError("Transaction columns must have equal lengths")
    if event_ids.ndim != 1 or event_timestamps.ndim != 1:
        raise ValueError("Event columns must be one-dimensional")
    if len(event_ids) != len(event_timestamps):
        raise ValueError("Event columns must have equal lengths")

    if tx_category is None:
        tx_category = np.zeros(len(tx_ids), dtype=np.int64)
    else:
        tx_category = np.ascontiguousarray(tx_category, dtype=np.int64)
        if tx_category.ndim != 1 or len(tx_category) != len(tx_ids):
            raise ValueError("Category column must match transaction rows")

    if len(event_ids) == 0:
        return np.empty(0, dtype=np.float64)

    order = np.lexsort((event_timestamps, event_ids))
    grouped_ids = event_ids[order]
    grouped_timestamps = event_timestamps[order]
    grouped_output = np.empty(len(event_ids), dtype=np.float64)
    _compute_grouped(
        tx_ids,
        tx_timestamps,
        tx_amount,
        tx_category,
        grouped_ids,
        grouped_timestamps,
        grouped_output,
        kind_code,
        window_code,
        window,
        0 if category_code is None else int(category_code),
    )

    output = np.empty(len(event_ids), dtype=np.float64)
    output[order] = grouped_output
    return output


__all__ = ["compute_event_feature"]

import numpy as np

from automatedfe.features.event_kernels import compute_event_feature
from automatedfe.features.feature_schema import DAY_MICROSECONDS


def test_time_window_two_pointers_preserve_exclusive_boundaries_and_event_order():
    transaction_ids = np.array([1, 1, 1, 1, 2, 2], dtype=np.int64)
    transaction_timestamps = np.array([10, 20, 20, 40, 5, 15], dtype=np.int64)
    transaction_amounts = np.ones(6, dtype=np.float64)

    # The input events are intentionally not grouped or timestamp-sorted.
    event_ids = np.array([2, 1, 1, 3, 1, 2], dtype=np.int64)
    event_timestamps = np.array([20, 45, 25, 100, 20, 5], dtype=np.int64)

    actual = compute_event_feature(
        transaction_ids,
        transaction_timestamps,
        transaction_amounts,
        None,
        event_ids,
        event_timestamps,
        kind="count_total",
        window_type="time",
        window=20,
    )

    # The interval is [event_timestamp - window, event_timestamp).
    np.testing.assert_array_equal(actual, [2, 1, 3, 0, 1, 0])


def test_day_window_two_pointers_handle_day_boundaries():
    day = DAY_MICROSECONDS
    transaction_ids = np.array([1, 1, 1, 1], dtype=np.int64)
    transaction_timestamps = np.array(
        [0, day, 2 * day, 2 * day + day // 2], dtype=np.int64
    )
    transaction_amounts = np.ones(4, dtype=np.float64)
    event_ids = np.array([1, 1, 1], dtype=np.int64)
    event_timestamps = np.array(
        [2 * day + day // 4, day + day // 2, 3 * day + day // 4], dtype=np.int64
    )

    actual = compute_event_feature(
        transaction_ids,
        transaction_timestamps,
        transaction_amounts,
        None,
        event_ids,
        event_timestamps,
        kind="count_total",
        window_type="days",
        window=2,
    )

    np.testing.assert_array_equal(actual, [2, 2, 2])

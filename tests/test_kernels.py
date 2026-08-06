import math

import numpy as np
import pytest

from automatedfe.features import (
    Aggregation,
    COUNT,
    MAX,
    MEAN,
    SUM,
    TIME_WINDOW,
    TOTAL_HISTORY,
    TotalHistoryWindow,
    RowWindow,
    TimeWindow,
    aggregate,
    sliding_window,
)

MERCHANT_IDS = np.array([1, 1, 1, 1, 2, 2, 3, 3, 3], dtype=np.int64)
AMOUNTS = np.array([10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 5.0, 5.0, 50.0])
CREATED_AT = np.array(
    [
        1000,
        2000,
        3000,
        10000,
        1000,
        2000,
        1000,
        1000,
        2000,
    ],
    dtype=np.int64,
)


def reference(merchant_id, values, timestamps, aggregation, window_mode, window_span):
    """Naive O(n * window) implementation of a trailing (exclusive) window."""

    out = []
    for i in range(len(merchant_id)):
        if window_mode == "rows":
            j = i - 1
            while (
                j >= 0
                and merchant_id[j] == merchant_id[i]
                and i - j <= window_span
            ):
                j -= 1
        elif window_mode == "time":
            j = i - 1
            while (
                j >= 0
                and merchant_id[j] == merchant_id[i]
                and timestamps[i] - timestamps[j] <= window_span
            ):
                j -= 1
        else:
            j = i - 1
            while j >= 0 and merchant_id[j] == merchant_id[i]:
                j -= 1
        window = values[j + 1 : i]

        if aggregation == "count":
            out.append(float(len(window)))
        elif aggregation == "sum":
            out.append(float(sum(window)) if window.size else math.nan)
        elif aggregation == "mean":
            out.append(sum(window) / len(window) if window.size else math.nan)
        elif aggregation == "max":
            out.append(float(max(window)) if window.size else math.nan)
        else:
            if window.size < 2:
                out.append(math.nan)
            else:
                mean = sum(window) / len(window)
                variance = sum((v - mean) ** 2 for v in window) / (len(window) - 1)
                out.append(math.sqrt(variance))
    return np.array(out)


@pytest.mark.parametrize("aggregation", ["count", "sum", "mean", "max"])
@pytest.mark.parametrize("window_span", [1, 2, 4, 1000])
def test_row_windows_match_reference(aggregation, window_span):
    actual = sliding_window(
        MERCHANT_IDS, AMOUNTS, None, aggregation=aggregation, window_mode="rows", window_span=window_span
    )
    expected = reference(MERCHANT_IDS, AMOUNTS, CREATED_AT, aggregation, "rows", window_span)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("aggregation", ["count", "sum", "mean", "max"])
@pytest.mark.parametrize("window_span", [1, 1000, 9000])
def test_time_windows_match_reference(aggregation, window_span):
    actual = sliding_window(
        MERCHANT_IDS, AMOUNTS, CREATED_AT, aggregation=aggregation, window_mode="time", window_span=window_span
    )
    expected = reference(MERCHANT_IDS, AMOUNTS, CREATED_AT, aggregation, "time", window_span)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("aggregation", ["count", "sum", "mean", "max"])
def test_total_history_matches_reference(aggregation):
    actual = sliding_window(
        MERCHANT_IDS, AMOUNTS, None, aggregation=aggregation, window_mode="total"
    )
    expected = reference(MERCHANT_IDS, AMOUNTS, CREATED_AT, aggregation, "total", 0)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_sliding_sum_subtracts_evicted_row():
    # 2-row window over [10, 20, 30, 40]: 10, 30, 50, 70.
    expected = np.array([math.nan, 10.0, 30.0, 50.0])
    actual = sliding_window(
        np.array([1, 1, 1, 1]),
        np.array([10.0, 20.0, 30.0, 40.0]),
        None,
        aggregation="sum",
        window_mode="rows",
        window_span=2,
    )
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_row_ring_buffer_wraps_repeatedly(aggregation):
    merchant_ids = np.ones(100, dtype=np.int64)
    values = np.arange(100, dtype=np.float64)
    actual = sliding_window(
        merchant_ids,
        values,
        None,
        aggregation=aggregation,
        window_mode="rows",
        window_span=3,
    )
    expected = reference(
        merchant_ids, values, np.empty(100), aggregation, "rows", 3
    )
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_maximum_deque_grows_beyond_initial_capacity():
    # Decreasing values keep every index in the monotonic deque, exercising
    # the same dynamic growth path used by the gp-benchmarks max kernels.
    rows = 1_500
    actual = sliding_window(
        np.ones(rows, dtype=np.int64),
        np.arange(rows, 0, -1, dtype=np.float64),
        None,
        aggregation="max",
        window_mode="total",
    )
    expected = np.full(rows, float(rows))
    expected[0] = math.nan
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_count_ignores_values():
    actual = sliding_window(
        np.array([1, 1, 2]),
        None,
        None,
        aggregation="count",
        window_mode="rows",
        window_span=5,
    )
    np.testing.assert_array_equal(actual, [0.0, 1.0, 0.0])


def test_aggregate_wrapper_with_feature_spec_types():
    actual = aggregate(
        MERCHANT_IDS,
        AMOUNTS,
        CREATED_AT,
        Aggregation.SUM,
        RowWindow(2),
    )
    expected = reference(MERCHANT_IDS, AMOUNTS, CREATED_AT, "sum", "rows", 2)
    np.testing.assert_allclose(actual, expected, equal_nan=True)

    time_actual = aggregate(
        MERCHANT_IDS, AMOUNTS, CREATED_AT, Aggregation.MEAN, TimeWindow(1000)
    )
    time_expected = reference(MERCHANT_IDS, AMOUNTS, CREATED_AT, "mean", "time", 1000)
    np.testing.assert_allclose(time_actual, time_expected, equal_nan=True)

    total_actual = aggregate(
        MERCHANT_IDS, AMOUNTS, None, Aggregation.MAX, TotalHistoryWindow()
    )
    total_expected = reference(MERCHANT_IDS, AMOUNTS, CREATED_AT, "max", "total", 0)
    np.testing.assert_allclose(total_actual, total_expected, equal_nan=True)


def test_duplicate_timestamps_kept_in_time_window():
    # Merchant 3 has two rows at t=1000; the equally-timed predecessor must
    # not be evicted when the next row arrives one microsecond-later.
    actual = sliding_window(
        np.array([3, 3, 3]),
        np.array([5.0, 5.0, 50.0]),
        np.array([1000, 1000, 2000]),
        aggregation="count",
        window_mode="time",
        window_span=1000,
    )
    np.testing.assert_array_equal(actual, [0.0, 1.0, 2.0])


def test_empty_input():
    actual = sliding_window(
        np.empty(0, dtype=np.int64), None, None, aggregation="count", window_mode="rows", window_span=5
    )
    assert actual.shape == (0,)


def test_aggregation_constants_match_strings():
    assert sliding_window(MERCHANT_IDS, AMOUNTS, None, aggregation="sum", window_mode="rows", window_span=2) is not None
    assert COUNT == 0
    assert SUM == 1
    assert MEAN == 2
    assert MAX == 3
    assert TIME_WINDOW == 1
    assert TOTAL_HISTORY == 2


@pytest.mark.parametrize("aggregation", ["sum", "mean", "max"])
def test_value_aggregations_require_values(aggregation):
    with pytest.raises(ValueError, match="requires the 'values' array"):
        sliding_window(
            MERCHANT_IDS, None, None, aggregation=aggregation, window_mode="rows", window_span=2
        )


def test_time_windows_require_timestamps():
    with pytest.raises(ValueError, match="require the 'timestamps' array"):
        sliding_window(
            MERCHANT_IDS, AMOUNTS, None, aggregation="sum", window_mode="time", window_span=1000
        )


@pytest.mark.parametrize("aggregation", ["median", "mode", "std"])
def test_unknown_aggregation_rejected(aggregation):
    with pytest.raises(ValueError, match="Unknown aggregation"):
        sliding_window(MERCHANT_IDS, None, None, aggregation=aggregation)


def test_unknown_window_mode_rejected():
    with pytest.raises(ValueError, match="Unknown window_mode"):
        sliding_window(MERCHANT_IDS, None, None, aggregation="count", window_mode="days")


def test_non_positive_span_rejected():
    with pytest.raises(ValueError, match="window_span"):
        sliding_window(MERCHANT_IDS, None, None, aggregation="count", window_mode="rows", window_span=0)


def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="rows, expected"):
        sliding_window(
            MERCHANT_IDS, np.array([1.0, 2.0]), None, aggregation="sum", window_mode="rows", window_span=2
        )

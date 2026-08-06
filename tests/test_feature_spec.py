from dataclasses import FrozenInstanceError

import pytest

from automatedfe.features import (
    ROW_WINDOWS,
    TIME_WINDOWS,
    TOTAL_HISTORY_WINDOW,
    TxFeature,
    RowWindow,
    TimeWindow,
    TotalHistoryWindow,
    WINDOW_CATALOG,
)


def test_row_windows_catalog():
    assert [window.rows for window in ROW_WINDOWS] == [
        5,
        10,
        20,
        50,
        100,
        200,
        400,
        800,
        1600,
    ]


def test_time_windows_catalog():
    assert [window.microseconds for window in TIME_WINDOWS] == [
        3_600_000_000,
        21_600_000_000,
        86_400_000_000,
        604_800_000_000,
        1_209_600_000_000,
        2_592_000_000_000,
        5_184_000_000_000,
        7_776_000_000_000,
    ]


def test_window_catalog_contents():
    assert TOTAL_HISTORY_WINDOW in WINDOW_CATALOG
    assert len(WINDOW_CATALOG) == 18
    assert all(isinstance(window, RowWindow) for window in ROW_WINDOWS)
    assert all(isinstance(window, TimeWindow) for window in TIME_WINDOWS)


@pytest.mark.parametrize("rows", [1, 10, 1000])
def test_row_window_positive(rows):
    assert RowWindow(rows).rows == rows


@pytest.mark.parametrize("rows", [0, -1, -100])
def test_row_window_rejects_non_positive(rows):
    with pytest.raises(ValueError, match="must be positive"):
        RowWindow(rows)


@pytest.mark.parametrize("microseconds", [1, 1000, 1_000_000])
def test_time_window_positive(microseconds):
    assert TimeWindow(microseconds).microseconds == microseconds


@pytest.mark.parametrize("microseconds", [0, -1, -100])
def test_time_window_rejects_non_positive(microseconds):
    with pytest.raises(ValueError, match="must be positive"):
        TimeWindow(microseconds)


def test_total_history_window():
    assert TotalHistoryWindow() is not None


def test_windows_are_immutable():
    window = RowWindow(10)
    with pytest.raises(FrozenInstanceError):
        window.rows = 20


def test_tx_feature_is_immutable():
    feature = TxFeature("mean", "amount", 10, "row")
    with pytest.raises(FrozenInstanceError):
        feature.window = 20


def test_tx_feature_names_are_deterministic():
    assert TxFeature("count_total", "amount", 20, "row").name == (
        "feat_count_total_amount_row_20"
    )
    assert TxFeature("mean", "amount", 604_800_000_000, "time").name == (
        "feat_mean_amount_time_7d"
    )
    assert TxFeature("std", "amount", 7_776_000_000_000, "time").name == (
        "feat_std_amount_time_90d"
    )
    assert TxFeature("mean", "amount", 21_600_000_000, "time").name == (
        "feat_mean_amount_time_6h"
    )
    assert TxFeature("max", "amount", -1, "time").name == (
        "feat_max_amount_time_total"
    )
    assert TxFeature(
        "count_category", "amount", 5, "row", 0, "status"
    ).name == "feat_count_category_status_0_amount_row_5"


def test_tx_feature_rejects_invalid_window():
    with pytest.raises(ValueError, match="positive"):
        TxFeature("mean", "amount", 0, "row")

    with pytest.raises(ValueError, match="Unsupported window type"):
        TxFeature("mean", "amount", 1, "weeks")

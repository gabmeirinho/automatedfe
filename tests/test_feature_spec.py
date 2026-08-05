from dataclasses import FrozenInstanceError

import pytest

from automatedfe.features import (
    Aggregation,
    FeatureSpec,
    ROW_WINDOWS,
    RowWindow,
    TIME_WINDOWS,
    TOTAL_HISTORY_WINDOW,
    TimeWindow,
    TotalHistoryWindow,
    WINDOW_CATALOG,
)


def test_row_windows_catalog():
    assert [w.rows for w in ROW_WINDOWS] == [5, 10, 20, 50, 100, 200, 400, 800, 1600]


def test_time_windows_catalog():
    assert [w.microseconds for w in TIME_WINDOWS] == [
        3_600_000_000,
        21_600_000_000,
        86_400_000_000,
        604_800_000_000,
        1_209_600_000_000,
        2_592_000_000_000,
        5_184_000_000_000,
        7_776_000_000_000,
    ]


def test_total_history_window_in_catalog():
    assert TOTAL_HISTORY_WINDOW in WINDOW_CATALOG


def test_window_catalog_contents():
    assert len(ROW_WINDOWS) == 9
    assert len(TIME_WINDOWS) == 8
    assert len(WINDOW_CATALOG) == 9 + 8 + 1
    assert all(isinstance(w, RowWindow) for w in ROW_WINDOWS)
    assert all(isinstance(w, TimeWindow) for w in TIME_WINDOWS)


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


def test_count_uses_none_input_column():
    spec = FeatureSpec(Aggregation.COUNT, None, RowWindow(10))
    assert spec.input_column is None


def test_other_aggregations_use_amount():
    for aggregation in (
        Aggregation.SUM,
        Aggregation.MEAN,
        Aggregation.MAX,
        Aggregation.STD,
    ):
        spec = FeatureSpec(aggregation, "amount", TotalHistoryWindow())
        assert spec.input_column == "amount"


@pytest.mark.parametrize(
    "aggregation, input_column",
    [
        (Aggregation.COUNT, "amount"),
        (Aggregation.SUM, None),
        (Aggregation.MEAN, "card_id"),
        (Aggregation.MAX, "card_id"),
        (Aggregation.STD, "card_id"),
    ],
)
def test_invalid_input_column_rejected(aggregation, input_column):
    with pytest.raises(ValueError, match="input_column"):
        FeatureSpec(aggregation, input_column, TotalHistoryWindow())


def test_windows_are_immutable():
    window = RowWindow(10)
    with pytest.raises(FrozenInstanceError):
        window.rows = 20


def test_feature_spec_is_immutable():
    spec = FeatureSpec(Aggregation.COUNT, None, RowWindow(10))
    with pytest.raises(FrozenInstanceError):
        spec.window = TotalHistoryWindow()

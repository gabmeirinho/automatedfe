from dataclasses import FrozenInstanceError

import pytest

from automatedfe.features import (
    Aggregation,
    FeatureSpec,
    RowWindow,
    TimeWindow,
    TotalHistoryWindow,
)


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

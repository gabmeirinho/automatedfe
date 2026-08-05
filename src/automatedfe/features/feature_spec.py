"""Typed, immutable definitions of feature transformations.

These dataclasses describe the building blocks of a genetic search over
transaction features. Each :class:`FeatureSpec` combines an
:class:`Aggregation`, an input column, and a window over which the
aggregation is computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

AMOUNT_COLUMN = "amount"

MICROSECONDS_PER_HOUR = 3_600_000_000


class Aggregation(str, Enum):
    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    STD = "std"


@dataclass(frozen=True, slots=True)
class RowWindow:
    """A window of the last *rows* transactions per merchant."""

    rows: int

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError(f"RowWindow.rows must be positive, got {self.rows}")

    @property
    def name(self) -> str:
        return f"last_{self.rows}_rows"


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A window of transactions in the last *microseconds* per merchant."""

    microseconds: int

    def __post_init__(self) -> None:
        if self.microseconds <= 0:
            raise ValueError(
                f"TimeWindow.microseconds must be positive, got {self.microseconds}"
            )

    @property
    def name(self) -> str:
        microseconds_per_hour = MICROSECONDS_PER_HOUR
        microseconds_per_day = 24 * microseconds_per_hour
        if self.microseconds % microseconds_per_day == 0:
            return f"last_{self.microseconds // microseconds_per_day}d"
        if self.microseconds % microseconds_per_hour == 0:
            return f"last_{self.microseconds // microseconds_per_hour}h"
        return f"last_{self.microseconds}us"


@dataclass(frozen=True, slots=True)
class TotalHistoryWindow:
    """A window covering the merchant's entire transaction history."""

    @property
    def name(self) -> str:
        return "all_history"


Window = RowWindow | TimeWindow | TotalHistoryWindow

ROW_WINDOWS: tuple[RowWindow, ...] = tuple(
    RowWindow(rows) for rows in (5, 10, 20, 50, 100, 200, 400, 800, 1600)
)

TIME_WINDOWS: tuple[TimeWindow, ...] = tuple(
    TimeWindow(hours * MICROSECONDS_PER_HOUR)
    for hours in (1, 6, 24, 24 * 7, 24 * 14, 24 * 30, 24 * 60, 24 * 90)
)

TOTAL_HISTORY_WINDOW = TotalHistoryWindow()

WINDOW_CATALOG: tuple[Window, ...] = (
    *ROW_WINDOWS,
    *TIME_WINDOWS,
    TOTAL_HISTORY_WINDOW,
)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """A single aggregation applied over a window.

    ``COUNT`` always counts transactions and therefore uses
    ``input_column=None``. All other aggregations summarize the
    ``"amount"`` column.
    """

    aggregation: Aggregation
    input_column: str | None
    window: Window

    def __post_init__(self) -> None:
        if self.aggregation is Aggregation.COUNT:
            if self.input_column is not None:
                raise ValueError(
                    "COUNT aggregation must use input_column=None "
                    f"(transaction count), got {self.input_column!r}"
                )
        elif self.input_column != AMOUNT_COLUMN:
            raise ValueError(
                f"{self.aggregation.value} aggregation must use "
                f"input_column={AMOUNT_COLUMN!r}, got {self.input_column!r}"
            )

    @property
    def name(self) -> str:
        column = "transactions" if self.aggregation is Aggregation.COUNT else self.input_column
        return f"{self.aggregation.value}_{column}_{self.window.name}"


__all__ = [
    "AMOUNT_COLUMN",
    "Aggregation",
    "FeatureSpec",
    "ROW_WINDOWS",
    "RowWindow",
    "TIME_WINDOWS",
    "TOTAL_HISTORY_WINDOW",
    "TimeWindow",
    "TotalHistoryWindow",
    "WINDOW_CATALOG",
    "Window",
]

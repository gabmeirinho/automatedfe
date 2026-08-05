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


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A window of transactions in the last *microseconds* per merchant."""

    microseconds: int

    def __post_init__(self) -> None:
        if self.microseconds <= 0:
            raise ValueError(
                f"TimeWindow.microseconds must be positive, got {self.microseconds}"
            )


@dataclass(frozen=True, slots=True)
class TotalHistoryWindow:
    """A window covering the merchant's entire transaction history."""


Window = RowWindow | TimeWindow | TotalHistoryWindow


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


__all__ = [
    "AMOUNT_COLUMN",
    "Aggregation",
    "FeatureSpec",
    "RowWindow",
    "TimeWindow",
    "TotalHistoryWindow",
    "Window",
]
